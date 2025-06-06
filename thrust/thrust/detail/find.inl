/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/adl/find.h>
#include <thrust/system/detail/generic/find.h>
#include <thrust/system/detail/generic/select_system.h>

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename T>
_CCCL_HOST_DEVICE InputIterator
find(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
     InputIterator first,
     InputIterator last,
     const T& value)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::find");
  using thrust::system::detail::generic::find;
  return find(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, value);
} // end find()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE InputIterator find_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::find_if");
  using thrust::system::detail::generic::find_if;
  return find_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end find_if()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE InputIterator find_if_not(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::find_if_not");
  using thrust::system::detail::generic::find_if_not;
  return find_if_not(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end find_if_not()

template <typename InputIterator, typename T>
InputIterator find(InputIterator first, InputIterator last, const T& value)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::find");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<InputIterator>::type;

  System system;

  return thrust::find(select_system(system), first, last, value);
}

template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first, InputIterator last, Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::find_if");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<InputIterator>::type;

  System system;

  return thrust::find_if(select_system(system), first, last, pred);
}

template <typename InputIterator, typename Predicate>
InputIterator find_if_not(InputIterator first, InputIterator last, Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::find_if_not");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<InputIterator>::type;

  System system;

  return thrust::find_if_not(select_system(system), first, last, pred);
}

THRUST_NAMESPACE_END
