//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_UTILITY_COPY
#define __CUDAX_UTILITY_COPY

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>

#include <cuda/experimental/__algorithm/common.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

namespace cuda::experimental
{

#if CUDA_VERSION >= 12080

template <typename _SrcTy, typename _DstTy>
void __copy_bytes_with_hints_impl(stream_ref __stream,
                      _CUDA_VSTD::span<_SrcTy> __src,
                      _CUDA_VSTD::span<_DstTy> __dst,
                      cudaMemcpySrcAccessOrder __access_time = cudaMemcpySrcAccessOrderStream)
{
  static_assert(!_CUDA_VSTD::is_const_v<_DstTy>, "Copy destination can't be const");
  static_assert(_CUDA_VSTD::is_trivially_copyable_v<_SrcTy> && _CUDA_VSTD::is_trivially_copyable_v<_DstTy>);

  if (__src.size_bytes() > __dst.size_bytes())
  {
    _CUDA_VSTD::__throw_invalid_argument("Copy destination is too small to fit the source data");
  }

  size_t __dummy_fail_idx = 0;
  size_t __another_zero = 0;
  void* __src_data = const_cast<void*>(static_cast<const void*>(__src.data()));
  void* __dst_data = __dst.data();
  size_t __copy_size = __src.size_bytes();

  cudaMemcpyAttributes __attributes{};
  __attributes.srcAccessOrder = __access_time;


  _CCCL_TRY_CUDA_API(
    ::cudaMemcpyBatchAsync,
    "Failed to perform a copy",
    &__dst_data,
    &__src_data,
    &__copy_size,
    1,
    &__attributes,
    &__another_zero,
    1,
    &__dummy_fail_idx,
    __stream.get());
}

#endif // CUDA_VERSION >= 12080

} // namespace cuda::experimental
#endif // __CUDAX_UTILITY_COPY
