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
// UNSUPPORTED: nvcc-12, nvcc-13.0, nvcc-13.1, nvcc-13.2
// UNSUPPORTED: pre-sm-80

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

#include <cuda/barrier>
#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

using barrier_t      = cuda::shared_barrier;
using cuda_barrier_t = cuda::barrier<cuda::thread_scope_block>;

static_assert(cuda::std::is_default_constructible_v<barrier_t>);
static_assert(!cuda::std::is_constructible_v<barrier_t, cuda::std::ptrdiff_t>);
static_assert(!cuda::std::is_constructible_v<barrier_t, cuda::shared_barrier_kind, cuda::std::ptrdiff_t>);
static_assert(!cuda::std::is_same_v<barrier_t::arrival_token, cuda::std::uint64_t>);
static_assert(!cuda::std::is_constructible_v<barrier_t::arrival_token, int>);
static_assert(!cuda::std::is_convertible_v<int, barrier_t::arrival_token>);
static_assert(!cuda::std::is_copy_constructible_v<barrier_t::operation_status>);
static_assert(!cuda::std::is_copy_assignable_v<barrier_t::operation_status>);
static_assert(cuda::std::is_move_constructible_v<barrier_t::operation_status>);
static_assert(cuda::std::is_move_assignable_v<barrier_t::operation_status>);
static_assert(!cuda::std::is_convertible_v<barrier_t::operation_status, bool>);
static_assert(barrier_t::max(cuda::shared_barrier_kind::completion_only) == (1 << 20) - 1);
static_assert(barrier_t::max(cuda::shared_barrier_kind::status_reporting) == (1 << 9) - 1);

template <class Barrier>
TEST_DEVICE_FUNC Barrier* construct_barrier(
  shared_memory_selector<Barrier, constructor_initializer>& sel, cuda::shared_barrier_kind kind, int expected)
{
  if constexpr (cuda::std::is_same_v<Barrier, barrier_t>)
  {
    Barrier* bar = sel.construct();
    execute_on_main_thread([&] {
      init(bar, kind, expected);
    });
    return bar;
  }
  else
  {
    unused(kind);
    return sel.construct(expected);
  }
}

template <class Barrier>
TEST_DEVICE_FUNC void check_barrier_kind(Barrier* bar, cuda::shared_barrier_kind kind)
{
  if constexpr (cuda::std::is_same_v<Barrier, barrier_t>)
  {
    assert(bar->is_kind(kind));
  }
  else
  {
    unused(bar, kind);
  }
}

template <class Barrier>
TEST_DEVICE_FUNC Barrier* construct_checked_barrier(
  shared_memory_selector<Barrier, constructor_initializer>& sel, cuda::shared_barrier_kind kind, int expected)
{
  Barrier* bar = construct_barrier(sel, kind, expected);
  __syncthreads();
  check_barrier_kind(bar, kind);
  return bar;
}

template <class Barrier>
TEST_DEVICE_FUNC void arrive_and_wait_ignoring_status(Barrier* bar)
{
  if constexpr (cuda::std::is_same_v<Barrier, barrier_t>)
  {
    bar->arrive_and_wait(cuda::ignore_status);
  }
  else
  {
    bar->arrive_and_wait();
  }
}

template <class Barrier>
TEST_DEVICE_FUNC void wait_token_ignoring_status(Barrier* bar, typename Barrier::arrival_token& token)
{
  if constexpr (cuda::std::is_same_v<Barrier, barrier_t>)
  {
    bar->wait(token, cuda::ignore_status);
  }
  else
  {
    bar->wait(cuda::std::move(token));
  }
}

template <class Barrier, class Duration>
TEST_DEVICE_FUNC bool
try_wait_token_for_ignoring_status(Barrier* bar, typename Barrier::arrival_token& token, Duration delay)
{
  if constexpr (cuda::std::is_same_v<Barrier, barrier_t>)
  {
    return bar->try_wait_for(token, delay, cuda::ignore_status);
  }
  else
  {
    return bar->try_wait_for(cuda::std::move(token), delay);
  }
}

template <class Barrier, class TimePoint>
TEST_DEVICE_FUNC bool
try_wait_token_until_ignoring_status(Barrier* bar, typename Barrier::arrival_token& token, TimePoint time)
{
  if constexpr (cuda::std::is_same_v<Barrier, barrier_t>)
  {
    return bar->try_wait_until(token, time, cuda::ignore_status);
  }
  else
  {
    return bar->try_wait_until(cuda::std::move(token), time);
  }
}

template <class Barrier>
TEST_DEVICE_FUNC void wait_phase(Barrier* bar, int phase)
{
  if constexpr (cuda::std::is_same_v<Barrier, barrier_t>)
  {
    bar->wait(phase, cuda::ignore_status);
  }
  else
  {
    bar->wait_parity(phase & 1);
  }
}

template <class Barrier, class Duration>
TEST_DEVICE_FUNC bool try_wait_phase_for(Barrier* bar, int phase, Duration delay)
{
  if constexpr (cuda::std::is_same_v<Barrier, barrier_t>)
  {
    return bar->try_wait_for(phase, delay, cuda::ignore_status);
  }
  else
  {
    return bar->try_wait_parity_for(phase & 1, delay);
  }
}

template <class Barrier, class TimePoint>
TEST_DEVICE_FUNC bool try_wait_phase_until(Barrier* bar, int phase, TimePoint time)
{
  if constexpr (cuda::std::is_same_v<Barrier, barrier_t>)
  {
    return bar->try_wait_until(phase, time, cuda::ignore_status);
  }
  else
  {
    return bar->try_wait_parity_until(phase & 1, time);
  }
}

template <class Barrier>
TEST_DEVICE_FUNC void test_concurrent_arrive_and_wait(cuda::shared_barrier_kind kind)
{
  shared_memory_selector<Barrier, constructor_initializer> sel;
  Barrier* bar = construct_checked_barrier(sel, kind, 2);

  auto worker = LAMBDA()
  {
    for (int i = 0; i != 10; ++i)
    {
      arrive_and_wait_ignoring_status(bar);
    }
  };

  concurrent_agents_launch(worker, worker);
}

template <class Barrier>
TEST_DEVICE_FUNC void test_concurrent_arrive_wait(cuda::shared_barrier_kind kind)
{
  shared_memory_selector<Barrier, constructor_initializer> sel;
  Barrier* bar = construct_checked_barrier(sel, kind, 2);

  typename Barrier::arrival_token* token = nullptr;
  execute_on_main_thread([&] {
    token = new auto(bar->arrive());
  });

  auto awaiter = LAMBDA()
  {
    wait_token_ignoring_status(bar, *token);
  };
  auto arriver = LAMBDA()
  {
    (void) bar->arrive();
  };

  concurrent_agents_launch(awaiter, arriver);

  execute_on_main_thread([&] {
    delete token;
    auto token2 = bar->arrive(2);
    wait_token_ignoring_status(bar, token2);
  });
}

template <class Barrier>
TEST_DEVICE_FUNC void test_concurrent_arrive_and_drop(cuda::shared_barrier_kind kind)
{
  shared_memory_selector<Barrier, constructor_initializer> sel;
  Barrier* bar = construct_checked_barrier(sel, kind, 2);

  auto dropper = LAMBDA()
  {
    bar->arrive_and_drop();
  };
  auto arriver = LAMBDA()
  {
    arrive_and_wait_ignoring_status(bar);
    arrive_and_wait_ignoring_status(bar);
  };

  concurrent_agents_launch(dropper, arriver);
}

template <class Barrier>
TEST_DEVICE_FUNC void test_concurrent_try_wait_for(cuda::shared_barrier_kind kind)
{
  shared_memory_selector<Barrier, constructor_initializer> sel;
  Barrier* bar = construct_checked_barrier(sel, kind, 2);
  cuda::std::chrono::nanoseconds delay(0);

  typename Barrier::arrival_token* token = nullptr;
  execute_on_main_thread([&] {
    token = new auto(bar->arrive());
  });

  auto awaiter = LAMBDA()
  {
    while (!try_wait_token_for_ignoring_status(bar, *token, delay))
    {
    }
  };
  auto arriver = LAMBDA()
  {
    (void) bar->arrive();
  };

  concurrent_agents_launch(awaiter, arriver);

  execute_on_main_thread([&] {
    delete token;
  });
}

template <class Barrier>
TEST_DEVICE_FUNC void test_concurrent_try_wait_until(cuda::shared_barrier_kind kind)
{
  shared_memory_selector<Barrier, constructor_initializer> sel;
  Barrier* bar = construct_checked_barrier(sel, kind, 2);
  cuda::std::chrono::duration<int> delay(0);

  typename Barrier::arrival_token* token = nullptr;
  execute_on_main_thread([&] {
    token = new auto(bar->arrive());
  });

  auto awaiter = LAMBDA()
  {
    auto until_time = cuda::std::chrono::system_clock::now() + delay;
    while (!try_wait_token_until_ignoring_status(bar, *token, until_time))
    {
    }
  };
  auto arriver = LAMBDA()
  {
    (void) bar->arrive();
  };

  concurrent_agents_launch(awaiter, arriver);

  execute_on_main_thread([&] {
    delete token;
  });
}

template <class Barrier>
TEST_DEVICE_FUNC void test_concurrent_wait_phase(cuda::shared_barrier_kind kind)
{
  shared_memory_selector<Barrier, constructor_initializer> sel;
  Barrier* bar = construct_checked_barrier(sel, kind, 2);
  int phase    = 0;

  execute_on_main_thread([&] {
    (void) bar->arrive();
  });

  auto awaiter = LAMBDA()
  {
    wait_phase(bar, phase);
  };
  auto arriver = LAMBDA()
  {
    (void) bar->arrive();
  };

  concurrent_agents_launch(awaiter, arriver);

  execute_on_main_thread([&] {
    auto token = bar->arrive(2);
    unused(token);
    wait_phase(bar, phase + 1);
  });
}

template <class Barrier>
TEST_DEVICE_FUNC void test_concurrent_try_wait_phase_for(cuda::shared_barrier_kind kind)
{
  shared_memory_selector<Barrier, constructor_initializer> sel;
  Barrier* bar = construct_checked_barrier(sel, kind, 2);
  int phase    = 0;
  cuda::std::chrono::nanoseconds delay(0);

  execute_on_main_thread([&] {
    (void) bar->arrive();
  });

  auto awaiter = LAMBDA()
  {
    while (!try_wait_phase_for(bar, phase, delay))
    {
    }
  };
  auto arriver = LAMBDA()
  {
    (void) bar->arrive();
  };

  concurrent_agents_launch(awaiter, arriver);
}

template <class Barrier>
TEST_DEVICE_FUNC void test_concurrent_try_wait_phase_until(cuda::shared_barrier_kind kind)
{
  shared_memory_selector<Barrier, constructor_initializer> sel;
  Barrier* bar = construct_checked_barrier(sel, kind, 2);
  int phase    = 0;
  cuda::std::chrono::duration<int> delay(0);

  execute_on_main_thread([&] {
    (void) bar->arrive();
  });

  auto awaiter = LAMBDA()
  {
    auto until_time = cuda::std::chrono::system_clock::now() + delay;
    while (!try_wait_phase_until(bar, phase, until_time))
    {
    }
  };
  auto arriver = LAMBDA()
  {
    (void) bar->arrive();
  };

  concurrent_agents_launch(awaiter, arriver);
}

template <class Barrier>
TEST_DEVICE_FUNC void test_shared_memory_barrier_choreography(cuda::shared_barrier_kind kind)
{
  test_concurrent_arrive_and_wait<Barrier>(kind);
  test_concurrent_arrive_wait<Barrier>(kind);
  test_concurrent_arrive_and_drop<Barrier>(kind);
  test_concurrent_try_wait_for<Barrier>(kind);
  test_concurrent_try_wait_until<Barrier>(kind);
  test_concurrent_wait_phase<Barrier>(kind);
  test_concurrent_try_wait_phase_for<Barrier>(kind);
  test_concurrent_try_wait_phase_until<Barrier>(kind);
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
  execute_on_main_thread([&] {
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

  execute_on_main_thread([&] {
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

TEST_DEVICE_FUNC void test_shared_barrier_common_extensions(cuda::shared_barrier_kind kind)
{
  shared_memory_selector<barrier_t, constructor_initializer> sel;
  barrier_t* bar = construct_checked_barrier(sel, kind, blockDim.x);

  test_test_waits(bar);
  test_ignore_status_waits(bar);
  test_status_waits(bar);
  test_phase_waits(bar);
}

TEST_DEVICE_FUNC void test_shared_barrier_sm90_extensions(cuda::shared_barrier_kind kind)
{
  shared_memory_selector<barrier_t, constructor_initializer> sel;
  barrier_t* bar = construct_checked_barrier(sel, kind, blockDim.x);

  test_tx_waits(bar);
}

TEST_DEVICE_FUNC void test_shared_barrier_device()
{
  test_shared_memory_barrier_choreography<cuda_barrier_t>(cuda::shared_barrier_kind::completion_only);
  test_shared_memory_barrier_choreography<barrier_t>(cuda::shared_barrier_kind::completion_only);
  test_shared_barrier_common_extensions(cuda::shared_barrier_kind::completion_only);

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (test_shared_memory_barrier_choreography<barrier_t>(cuda::shared_barrier_kind::status_reporting);
                test_shared_barrier_common_extensions(cuda::shared_barrier_kind::status_reporting);
                test_shared_barrier_sm90_extensions(cuda::shared_barrier_kind::status_reporting);))
#endif // __cccl_ptx_isa >= 940
  NV_IF_TARGET(NV_PROVIDES_SM_90, (test_shared_barrier_sm90_extensions(cuda::shared_barrier_kind::completion_only);))
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (
      // Required by concurrent_agents_launch to know how many threads to launch.
      cuda_thread_count = 2;),
    NV_IS_DEVICE,
    (test_shared_barrier_device();))

  return 0;
}
