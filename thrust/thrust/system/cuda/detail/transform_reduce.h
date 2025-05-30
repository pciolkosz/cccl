/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CUDA_COMPILER()
#  include <thrust/system/cuda/config.h>

#  include <cub/device/device_reduce.cuh>
#  include <cub/util_math.cuh>

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/raw_reference_cast.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/distance.h>
#  include <thrust/functional.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/core/agent_launcher.h>
#  include <thrust/system/cuda/detail/dispatch.h>
#  include <thrust/system/cuda/detail/get_value.h>
#  include <thrust/system/cuda/detail/make_unsigned_special.h>
#  include <thrust/system/cuda/detail/par_to_seq.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{

namespace detail
{

template <typename Derived, typename InputIt, typename Size, typename UnaryOp, typename T, typename BinaryOp>
THRUST_RUNTIME_FUNCTION T transform_reduce_n_impl(
  execution_policy<Derived>& policy, InputIt first, Size num_items, UnaryOp unary_op, T init, BinaryOp binary_op)
{
  cudaStream_t stream = cuda_cub::stream(policy);
  cudaError_t status;

  // Determine temporary device storage requirements.

  size_t tmp_size = 0;

  THRUST_INDEX_TYPE_DISPATCH(
    status,
    cub::DeviceReduce::TransformReduce,
    num_items,
    (nullptr, tmp_size, first, static_cast<T*>(nullptr), num_items_fixed, binary_op, unary_op, init, stream));
  cuda_cub::throw_on_error(status, "after reduction step 1");

  // Allocate temporary storage.

  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, sizeof(T) + tmp_size);

  // Run reduction.

  // `tmp.begin()` yields a `normal_iterator`, which dereferences to a
  // `reference`, which has an `operator&` that returns a `pointer`, which
  // has a `.get` method that returns a raw pointer, which we can (finally)
  // `static_cast` to `void*`.
  //
  // The array was dynamically allocated, so we assume that it's suitably
  // aligned for any type of data. `malloc`/`cudaMalloc`/`new`/`std::allocator`
  // make this guarantee.
  T* ret_ptr    = thrust::detail::aligned_reinterpret_cast<T*>(tmp.data().get());
  void* tmp_ptr = static_cast<void*>((tmp.data() + sizeof(T)).get());
  THRUST_INDEX_TYPE_DISPATCH(
    status,
    cub::DeviceReduce::TransformReduce,
    num_items,
    (tmp_ptr, tmp_size, first, ret_ptr, num_items_fixed, binary_op, unary_op, init, stream));
  cuda_cub::throw_on_error(status, "after reduction step 2");

  // Synchronize the stream and get the value.

  status = cuda_cub::synchronize(policy);
  cuda_cub::throw_on_error(status, "reduce failed to synchronize");

  // `tmp.begin()` yields a `normal_iterator`, which dereferences to a
  // `reference`, which has an `operator&` that returns a `pointer`, which
  // has a `.get` method that returns a raw pointer, which we can (finally)
  // `static_cast` to `void*`.
  //
  // The array was dynamically allocated, so we assume that it's suitably
  // aligned for any type of data. `malloc`/`cudaMalloc`/`new`/`std::allocator`
  // make this guarantee.
  return thrust::cuda_cub::get_value(policy, thrust::detail::aligned_reinterpret_cast<T*>(tmp.data().get()));
}

} // namespace detail

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class TransformOp, class T, class ReduceOp>
T _CCCL_HOST_DEVICE transform_reduce(
  execution_policy<Derived>& policy, InputIt first, InputIt last, TransformOp transform_op, T init, ReduceOp reduce_op)
{
  using size_type           = thrust::detail::it_difference_t<InputIt>;
  const size_type num_items = static_cast<size_type>(::cuda::std::distance(first, last));

  THRUST_CDP_DISPATCH(
    (init = thrust::cuda_cub::detail::transform_reduce_n_impl(policy, first, num_items, transform_op, init, reduce_op);),
    (init = thrust::transform_reduce(
       cvt_to_seq(derived_cast(policy)), first, first + num_items, transform_op, init, reduce_op);));
  return init;
}

} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
