//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_EXAMPLES_JIT_DISPATCH_NVRTC_KERNEL_CUH
#define CUDAX_EXAMPLES_JIT_DISPATCH_NVRTC_KERNEL_CUH

#include <cuda/experimental/__kernel/jit_dispatch.cuh>

namespace cudax_example::jit_dispatch_nvrtc
{
struct small_copy_impl
{
  static constexpr ::cuda::experimental::prototype::jit_kernel_id id = 11;

  _CCCL_DEVICE_API void operator()(int* dst, const int* src, int count) const
#if !defined(CUDAX_EXAMPLE_JIT_DISPATCH_USE_NVRTC) || defined(__CUDACC_RTC__)
  {
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      for (int i = 0; i != count; ++i)
      {
        dst[i] = src[i];
      }
    }
  }
#else
  ;
#endif
};

struct scalar_copy_impl
{
  static constexpr ::cuda::experimental::prototype::jit_kernel_id id = 37;

  _CCCL_DEVICE_API void operator()(int* dst, const int* src, int count) const
#if !defined(CUDAX_EXAMPLE_JIT_DISPATCH_USE_NVRTC) || defined(__CUDACC_RTC__)
  {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride)
    {
      dst[i] = src[i];
    }
  }
#else
  ;
#endif
};

struct unrolled_copy_2_impl
{
  static constexpr ::cuda::experimental::prototype::jit_kernel_id id = 41;

  _CCCL_DEVICE_API void operator()(int* dst, const int* src, int count) const
#if !defined(CUDAX_EXAMPLE_JIT_DISPATCH_USE_NVRTC) || defined(__CUDACC_RTC__)
  {
    const int stride = blockDim.x * gridDim.x * 2;
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2; i < count; i += stride)
    {
      dst[i] = src[i];
      if (i + 1 < count)
      {
        dst[i + 1] = src[i + 1];
      }
    }
  }
#else
  ;
#endif
};

struct unrolled_copy_4_impl
{
  static constexpr ::cuda::experimental::prototype::jit_kernel_id id = 43;

  _CCCL_DEVICE_API void operator()(int* dst, const int* src, int count) const
#if !defined(CUDAX_EXAMPLE_JIT_DISPATCH_USE_NVRTC) || defined(__CUDACC_RTC__)
  {
    const int stride = blockDim.x * gridDim.x * 4;
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4; i < count; i += stride)
    {
      dst[i] = src[i];
      if (i + 1 < count)
      {
        dst[i + 1] = src[i + 1];
      }
      if (i + 2 < count)
      {
        dst[i + 2] = src[i + 2];
      }
      if (i + 3 < count)
      {
        dst[i + 3] = src[i + 3];
      }
    }
  }
#else
  ;
#endif
};

using copy_kernel = ::cuda::experimental::prototype::
  jit_kernel<small_copy_impl, scalar_copy_impl, unrolled_copy_2_impl, unrolled_copy_4_impl>;
using dispatcher_type = copy_kernel::dispatcher_type;
} // namespace cudax_example::jit_dispatch_nvrtc

#if !defined(__CUDACC_RTC__)
template <>
struct cuda::experimental::prototype::jit_kernel_source<cudax_example::jit_dispatch_nvrtc::dispatcher_type>
{
  static constexpr const char* implementation_header = "jit_dispatch_nvrtc_kernel.cuh";
  static constexpr const char* dispatcher_type_name  = "cudax_example::jit_dispatch_nvrtc::dispatcher_type";
  inline static constexpr const char* include_paths[] = {
    CUDAX_EXAMPLE_CUDAX_INCLUDE_DIR,
    CUDAX_EXAMPLE_LIBCUDACXX_INCLUDE_DIR,
    CUDAX_EXAMPLE_DIR,
  };
  static constexpr ::cuda::std::size_t include_path_count = sizeof(include_paths) / sizeof(include_paths[0]);
};
#endif // !defined(__CUDACC_RTC__)

#endif // CUDAX_EXAMPLES_JIT_DISPATCH_NVRTC_KERNEL_CUH
