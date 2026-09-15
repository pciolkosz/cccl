//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
// UNSUPPORTED: nvcc-12, nvcc-13.0, nvcc-13.1, nvcc-13.2, nvcc-13.3
// UNSUPPORTED: pre-sm-90

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

#include <cuda/barrier>
#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <new> // IWYU pragma: keep (needed for placement new)

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

using barrier_t = cuda::shared_barrier;

static_assert(cuda::std::is_default_constructible_v<barrier_t>);
static_assert(!cuda::std::is_constructible_v<barrier_t, cuda::std::ptrdiff_t>);
static_assert(!cuda::std::is_same_v<barrier_t::arrival_token, cuda::std::uint64_t>);
static_assert(cuda::std::is_default_constructible_v<barrier_t::arrival_token>);
static_assert(!cuda::std::is_constructible_v<barrier_t::arrival_token, int>);
static_assert(!cuda::std::is_convertible_v<int, barrier_t::arrival_token>);
static_assert(!cuda::std::is_copy_constructible_v<barrier_t::operation_status>);
static_assert(!cuda::std::is_copy_assignable_v<barrier_t::operation_status>);
static_assert(cuda::std::is_move_constructible_v<barrier_t::operation_status>);
static_assert(cuda::std::is_move_assignable_v<barrier_t::operation_status>);
static_assert(!cuda::std::is_convertible_v<barrier_t::operation_status, bool>);
static_assert(barrier_t::max() == (1 << 9) - 1);

template <class Fn>
TEST_DEVICE_FUNC void execute_on_thread_zero(Fn&& fn)
{
  if (threadIdx.x == 0)
  {
    fn();
  }
  __syncthreads();
}

template <class Fn0, class Fn1>
TEST_DEVICE_FUNC void concurrent_threads_launch(Fn0 fn0, Fn1 fn1)
{
  assert(blockDim.x == 2);

  __syncthreads();

  if (threadIdx.x == 0)
  {
    fn0();
  }
  else if (threadIdx.x == 1)
  {
    fn1();
  }

  __syncthreads();
}

TEST_DEVICE_FUNC barrier_t* construct_barrier(int expected)
{
  alignas(barrier_t) __shared__ char storage[sizeof(barrier_t)];
  barrier_t* bar = reinterpret_cast<barrier_t*>(storage);
  execute_on_thread_zero([&] {
    new ((void*) bar) barrier_t;
    init(bar, expected);
  });
  return bar;
}

struct shared_barrier_owner
{
  barrier_t* bar = nullptr;

  TEST_DEVICE_FUNC explicit shared_barrier_owner(int expected)
      : bar(construct_barrier(expected))
  {
    __syncthreads();
  }

  TEST_DEVICE_FUNC ~shared_barrier_owner()
  {
    __syncthreads();
    execute_on_thread_zero([&] {
      bar->~barrier_t();
    });
    __syncthreads();
  }
};

TEST_DEVICE_FUNC void test_concurrent_arrive_and_wait()
{
  shared_barrier_owner barrier(2);
  barrier_t* bar = barrier.bar;

  auto worker = [=] __device__ {
    for (int i = 0; i != 10; ++i)
    {
      bar->arrive_and_wait(cuda::ignore_status);
    }
  };

  concurrent_threads_launch(worker, worker);
}

TEST_DEVICE_FUNC void test_concurrent_arrive_wait()
{
  shared_barrier_owner barrier(2);
  barrier_t* bar = barrier.bar;

  barrier_t::arrival_token* token = nullptr;
  execute_on_thread_zero([&] {
    token = new auto(bar->arrive());
  });

  auto awaiter = [=] __device__ {
    bar->wait(*token, cuda::ignore_status);
  };
  auto arriver = [=] __device__ {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);

  execute_on_thread_zero([&] {
    delete token;
    auto token2 = bar->arrive(2);
    bar->wait(token2, cuda::ignore_status);
  });
}

TEST_DEVICE_FUNC void test_concurrent_arrive_and_drop()
{
  shared_barrier_owner barrier(2);
  barrier_t* bar = barrier.bar;

  auto dropper = [=] __device__ {
    bar->arrive_and_drop();
  };
  auto arriver = [=] __device__ {
    bar->arrive_and_wait(cuda::ignore_status);
    bar->arrive_and_wait(cuda::ignore_status);
  };

  concurrent_threads_launch(dropper, arriver);
}

TEST_DEVICE_FUNC void test_concurrent_try_wait_for()
{
  shared_barrier_owner barrier(2);
  barrier_t* bar = barrier.bar;
  cuda::std::chrono::nanoseconds delay(0);

  barrier_t::arrival_token* token = nullptr;
  execute_on_thread_zero([&] {
    token = new auto(bar->arrive());
  });

  auto awaiter = [=] __device__ {
    while (!bar->try_wait_for(*token, delay, cuda::ignore_status))
    {
    }
  };
  auto arriver = [=] __device__ {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);

  execute_on_thread_zero([&] {
    delete token;
  });
}

TEST_DEVICE_FUNC void test_concurrent_try_wait_until()
{
  shared_barrier_owner barrier(2);
  barrier_t* bar = barrier.bar;
  cuda::std::chrono::duration<int> delay(0);

  barrier_t::arrival_token* token = nullptr;
  execute_on_thread_zero([&] {
    token = new auto(bar->arrive());
  });

  auto awaiter = [=] __device__ {
    auto until_time = cuda::std::chrono::system_clock::now() + delay;
    while (!bar->try_wait_until(*token, until_time, cuda::ignore_status))
    {
    }
  };
  auto arriver = [=] __device__ {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);

  execute_on_thread_zero([&] {
    delete token;
  });
}

TEST_DEVICE_FUNC void test_concurrent_wait_phase()
{
  shared_barrier_owner barrier(2);
  barrier_t* bar            = barrier.bar;
  cuda::std::uint32_t phase = 0;

  execute_on_thread_zero([&] {
    (void) bar->arrive();
  });

  auto awaiter = [=] __device__ {
    bar->wait(phase, cuda::ignore_status);
  };
  auto arriver = [=] __device__ {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);

  execute_on_thread_zero([&] {
    auto token = bar->arrive(2);
    unused(token);
    bar->wait(phase + 1, cuda::ignore_status);
  });
}

TEST_DEVICE_FUNC void test_concurrent_try_wait_phase_for()
{
  shared_barrier_owner barrier(2);
  barrier_t* bar            = barrier.bar;
  cuda::std::uint32_t phase = 0;
  cuda::std::chrono::nanoseconds delay(0);

  execute_on_thread_zero([&] {
    (void) bar->arrive();
  });

  auto awaiter = [=] __device__ {
    while (!bar->try_wait_for(phase, delay, cuda::ignore_status))
    {
    }
  };
  auto arriver = [=] __device__ {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);
}

TEST_DEVICE_FUNC void test_concurrent_try_wait_phase_until()
{
  shared_barrier_owner barrier(2);
  barrier_t* bar            = barrier.bar;
  cuda::std::uint32_t phase = 0;
  cuda::std::chrono::duration<int> delay(0);

  execute_on_thread_zero([&] {
    (void) bar->arrive();
  });

  auto awaiter = [=] __device__ {
    auto until_time = cuda::std::chrono::system_clock::now() + delay;
    while (!bar->try_wait_until(phase, until_time, cuda::ignore_status))
    {
    }
  };
  auto arriver = [=] __device__ {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);
}

TEST_DEVICE_FUNC void test_shared_memory_barrier_choreography()
{
  test_concurrent_arrive_and_wait();
  test_concurrent_arrive_wait();
  test_concurrent_arrive_and_drop();
  test_concurrent_try_wait_for();
  test_concurrent_try_wait_until();
  test_concurrent_wait_phase();
  test_concurrent_try_wait_phase_for();
  test_concurrent_try_wait_phase_until();
}

TEST_DEVICE_FUNC void check_success_status(const barrier_t::operation_status& status)
{
  assert(status.complete());
  assert(!status.has_report());

  assert(status.get_error_count(cuda::status_source::generic_fabric) == 0);

  unsigned int visited = 0;
  status.for_each_error(cuda::status_source::generic_fabric, [&](cudaFabricOpStatusInfo) {
    ++visited;
  });
  assert(visited == 0);
}

TEST_DEVICE_FUNC void complete_tx(barrier_t& bar, int transaction_count)
{
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_90,
    (cuda::ptx::mbarrier_complete_tx(
       cuda::ptx::sem_relaxed,
       cuda::ptx::scope_cta,
       cuda::ptx::space_shared,
       bar.native_handle(),
       static_cast<::cuda::std::uint32_t>(transaction_count));
     return;),
    NV_ANY_TARGET,
    (__trap();));
}

TEST_DEVICE_FUNC void test_tx_wait(barrier_t* bar, int tx_count)
{
  auto token = bar->arrive_tx(1, tx_count);
  complete_tx(*bar, tx_count);
  bar->wait(token, cuda::ignore_status);

  bar->expect_tx(tx_count);
  token = bar->arrive();
  complete_tx(*bar, tx_count);
  bar->wait(token, cuda::ignore_status);

  token = bar->arrive_tx(1, tx_count);
  complete_tx(*bar, tx_count);
  auto status = bar->wait(token, cuda::return_status);
  check_success_status(status);
}

TEST_DEVICE_FUNC void test_tx_waits(barrier_t* bar)
{
  test_tx_wait(bar, 1);
  test_tx_wait(bar, 1024);
}

TEST_DEVICE_FUNC void test_test_waits(barrier_t* bar)
{
  barrier_t::arrival_token* token = nullptr;
  execute_on_thread_zero([&] {
    token = new auto(bar->arrive());
    assert(!bar->test_wait(*token, cuda::ignore_status));

    auto status = bar->test_wait(*token, cuda::return_status);
    assert(!status.complete());
    assert(!status.has_report());
  });

  __syncthreads();

  if (threadIdx.x != 0)
  {
    unused(bar->arrive());
  }

  __syncthreads();

  execute_on_thread_zero([&] {
    assert(bar->test_wait(*token, cuda::ignore_status));

    auto status = bar->test_wait(*token, cuda::return_status);
    check_success_status(status);
    delete token;
  });

  __syncthreads();
}

TEST_DEVICE_FUNC void test_ignore_status_waits(barrier_t* bar)
{
  auto token = bar->arrive();
  while (!bar->try_wait(token, cuda::ignore_status))
  {
  }

  token = bar->arrive();
  bar->wait(token, cuda::ignore_status);

  bar->arrive_and_wait(cuda::ignore_status);
}

TEST_DEVICE_FUNC void test_status_waits(barrier_t* bar)
{
  auto token       = bar->arrive();
  auto poll_status = bar->try_wait(token, cuda::return_status);
  if (poll_status.complete())
  {
    check_success_status(poll_status);
  }

  auto status = bar->wait(token, cuda::return_status);
  check_success_status(status);

  status = bar->arrive_and_wait(cuda::return_status);
  check_success_status(status);

  bar->arrive_and_wait(cuda::ignore_status);
}

TEST_DEVICE_FUNC void test_phase_waits(barrier_t* bar)
{
  auto token = bar->arrive();
  unused(token);
  bar->wait(0, cuda::ignore_status);

  assert(bar->test_wait(0, cuda::ignore_status));

  auto status = bar->test_wait(0, cuda::return_status);
  check_success_status(status);

  bar->wait(0, cuda::ignore_status);

  status = bar->wait(0, cuda::return_status);
  check_success_status(status);

  bar->wait_conditional_phase(0);
  assert(bar->test_wait_conditional_phase(0));
  assert(bar->try_wait_conditional_phase(0));

  bar->arrive_and_wait(cuda::ignore_status);

  bar->wait(1, cuda::ignore_status);

  status = bar->wait(1, cuda::return_status);
  check_success_status(status);

  bar->arrive_and_wait(cuda::ignore_status);

  assert(bar->try_wait_for(2, cuda::std::chrono::nanoseconds(1), cuda::ignore_status));

  status = bar->try_wait_for(2, cuda::std::chrono::nanoseconds(1), cuda::return_status);
  check_success_status(status);

  bar->arrive_and_wait(cuda::ignore_status);

  assert(bar->try_wait_until(
    3, cuda::std::chrono::system_clock::now() + cuda::std::chrono::seconds(1), cuda::ignore_status));

  status =
    bar->try_wait_until(3, cuda::std::chrono::system_clock::now() + cuda::std::chrono::seconds(1), cuda::return_status);
  check_success_status(status);
}

TEST_DEVICE_FUNC void test_shared_barrier_common_extensions()
{
  shared_barrier_owner barrier(blockDim.x);
  barrier_t* bar = barrier.bar;

  test_test_waits(bar);
  test_ignore_status_waits(bar);
  test_status_waits(bar);
  test_phase_waits(bar);
}

TEST_DEVICE_FUNC void test_shared_barrier_sm90_extensions()
{
  shared_barrier_owner barrier(blockDim.x);
  barrier_t* bar = barrier.bar;

  test_tx_waits(bar);
}

TEST_DEVICE_FUNC void test_shared_barrier_device()
{
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (test_shared_memory_barrier_choreography(); test_shared_barrier_common_extensions();
                test_shared_barrier_sm90_extensions();))
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (
      // Required by the device-side concurrent_threads_launch helper.
      cuda_thread_count = 2;),
    NV_IS_DEVICE,
    (test_shared_barrier_device();))

  return 0;
}
