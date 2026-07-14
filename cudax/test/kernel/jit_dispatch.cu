//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/kernel.cuh>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include <testing.cuh>

namespace
{
namespace jit = cuda::experimental::prototype;

struct payload
{
  alignas(16) int values[4];
};

struct add_impl
{
  static constexpr jit::jit_kernel_id id = 7;
  using packed_tuple_type               = jit::jit_packed_tuple<int*, int>;

  _CCCL_HOST_DEVICE void operator()(int* out, int value) const
  {
    *out += value;
  }
};

struct add_twice_impl
{
  static constexpr jit::jit_kernel_id id = 9;

  _CCCL_HOST_DEVICE void operator()(int* out, int value) const
  {
    *out += value * 2;
  }
};

struct payload_impl
{
  static constexpr jit::jit_kernel_id id = 42;
  using packed_tuple_type                = jit::jit_packed_tuple<int*, int, payload>;

  _CCCL_HOST_DEVICE void operator()(int* out, int scale, payload data) const
  {
    *out += scale * data.values[0];
  }
};

using test_jit_kernel      = jit::jit_kernel<add_impl, payload_impl>;
using same_args_jit_kernel = jit::jit_kernel<add_impl, add_twice_impl>;

__global__ void cccl_jit_dispatch_entry(test_jit_kernel::dispatcher_type dispatcher, bool* matched)
{
  *matched = test_jit_kernel::dispatch(dispatcher);
}
} // namespace

C2H_CCCLRT_TEST("JIT dispatch prototype packs arguments into type-erased dispatchers", "[jit_dispatch]")
{
  STATIC_REQUIRE(test_jit_kernel::max_packed_size == sizeof(payload_impl::packed_tuple_type));
  STATIC_REQUIRE(test_jit_kernel::max_packed_align == alignof(payload_impl::packed_tuple_type));
  STATIC_REQUIRE(test_jit_kernel::contains_implementation_v<add_impl>);
  STATIC_REQUIRE(test_jit_kernel::contains_implementation_v<payload_impl>);
  STATIC_REQUIRE(
    cuda::std::is_same_v<test_jit_kernel::dispatcher_type, jit::jit_dispatcher<add_impl, payload_impl>>);
  STATIC_REQUIRE(same_args_jit_kernel::max_packed_size == sizeof(jit::jit_packed_tuple<int*, int>));
  STATIC_REQUIRE(cuda::std::is_trivially_copyable_v<test_jit_kernel::dispatcher_type>);
  STATIC_REQUIRE(cuda::std::is_trivially_copyable_v<add_impl::packed_tuple_type>);
  STATIC_REQUIRE(cuda::std::is_trivially_copyable_v<payload_impl::packed_tuple_type>);

  int value = 1;
  auto add_dispatcher = test_jit_kernel::make_dispatcher<add_impl::id>(&value, 4);

  REQUIRE(add_dispatcher.id() == add_impl::id);
  REQUIRE(test_jit_kernel::dispatch(add_dispatcher));
  REQUIRE(value == 5);

  payload data{{3, 0, 0, 0}};
  auto payload_dispatcher = test_jit_kernel::make_dispatcher<payload_impl::id>(&value, 2, data);

  REQUIRE(payload_dispatcher.id() == payload_impl::id);
  REQUIRE(test_jit_kernel::dispatch(payload_dispatcher));
  REQUIRE(value == 11);

  auto runtime_dispatcher = same_args_jit_kernel::make_dispatcher(add_twice_impl::id, &value, 3);

  REQUIRE(runtime_dispatcher.id() == add_twice_impl::id);
  REQUIRE(same_args_jit_kernel::dispatch(runtime_dispatcher));
  REQUIRE(value == 17);

  [[maybe_unused]] auto entry = cccl_jit_dispatch_entry;
}
