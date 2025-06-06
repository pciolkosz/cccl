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

#include <thrust/system/cuda/config.h>

#include <thrust/advance.h>
#include <thrust/system/cuda/detail/cdp_dispatch.h>
#include <thrust/system/cuda/detail/cross_system.h>
#include <thrust/system/cuda/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy, typename InputIt, typename OutputIt>
_CCCL_HOST_DEVICE OutputIt
copy(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, InputIt first, InputIt last, OutputIt result);

template <class DerivedPolicy, class InputIt, class Size, class OutputIt>
_CCCL_HOST_DEVICE OutputIt
copy_n(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, InputIt first, Size n, OutputIt result);

namespace cuda_cub
{

// D->D copy requires NVCC compiler
template <class System, class InputIterator, class OutputIterator>
OutputIterator _CCCL_HOST_DEVICE
copy(execution_policy<System>& system, InputIterator first, InputIterator last, OutputIterator result);

template <class System1, class System2, class InputIterator, class OutputIterator>
OutputIterator _CCCL_HOST
copy(cross_system<System1, System2> systems, InputIterator first, InputIterator last, OutputIterator result);

template <class System, class InputIterator, class Size, class OutputIterator>
OutputIterator _CCCL_HOST_DEVICE
copy_n(execution_policy<System>& system, InputIterator first, Size n, OutputIterator result);

template <class System1, class System2, class InputIterator, class Size, class OutputIterator>
OutputIterator _CCCL_HOST
copy_n(cross_system<System1, System2> systems, InputIterator first, Size n, OutputIterator result);

} // namespace cuda_cub
THRUST_NAMESPACE_END

#include <thrust/system/cuda/detail/internal/copy_cross_system.h>
#include <thrust/system/cuda/detail/internal/copy_device_to_device.h>
#include <thrust/system/cuda/detail/par_to_seq.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{

#if _CCCL_HAS_CUDA_COMPILER()
// D->D copy requires NVCC compiler

_CCCL_EXEC_CHECK_DISABLE
template <class System, class InputIterator, class OutputIterator>
OutputIterator _CCCL_HOST_DEVICE
copy(execution_policy<System>& system, InputIterator first, InputIterator last, OutputIterator result)
{
  THRUST_CDP_DISPATCH((result = __copy::device_to_device(system, first, last, result);),
                      (result = thrust::copy(cvt_to_seq(derived_cast(system)), first, last, result);));
  return result;
} // end copy()

_CCCL_EXEC_CHECK_DISABLE
template <class System, class InputIterator, class Size, class OutputIterator>
OutputIterator _CCCL_HOST_DEVICE
copy_n(execution_policy<System>& system, InputIterator first, Size n, OutputIterator result)
{
  THRUST_CDP_DISPATCH((result = __copy::device_to_device(system, first, ::cuda::std::next(first, n), result);),
                      (result = thrust::copy_n(cvt_to_seq(derived_cast(system)), first, n, result);));
  return result;
} // end copy_n()
#endif

template <class System1, class System2, class InputIterator, class OutputIterator>
OutputIterator _CCCL_HOST
copy(cross_system<System1, System2> systems, InputIterator first, InputIterator last, OutputIterator result)
{
  return __copy::cross_system_copy(systems, first, last, result);
} // end copy()

template <class System1, class System2, class InputIterator, class Size, class OutputIterator>
OutputIterator _CCCL_HOST
copy_n(cross_system<System1, System2> systems, InputIterator first, Size n, OutputIterator result)
{
  return __copy::cross_system_copy_n(systems, first, n, result);
} // end copy_n()

} // namespace cuda_cub
THRUST_NAMESPACE_END

#include <thrust/detail/temporary_array.h>
#include <thrust/memory.h>
