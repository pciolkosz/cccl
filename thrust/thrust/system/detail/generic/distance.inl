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
#include <thrust/system/detail/generic/distance.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{

_CCCL_EXEC_CHECK_DISABLE
template <typename InputIterator>
inline _CCCL_HOST_DEVICE thrust::detail::it_difference_t<InputIterator>
distance(InputIterator first, InputIterator last, thrust::incrementable_traversal_tag)
{
  thrust::detail::it_difference_t<InputIterator> result(0);

  while (first != last)
  {
    ++first;
    ++result;
  } // end while

  return result;
} // end advance()

_CCCL_EXEC_CHECK_DISABLE
template <typename InputIterator>
inline _CCCL_HOST_DEVICE thrust::detail::it_difference_t<InputIterator>
distance(InputIterator first, InputIterator last, thrust::random_access_traversal_tag)
{
  return last - first;
} // end distance()

} // namespace detail

_CCCL_EXEC_CHECK_DISABLE
template <typename InputIterator>
inline _CCCL_HOST_DEVICE thrust::detail::it_difference_t<InputIterator> distance(InputIterator first, InputIterator last)
{
  // dispatch on iterator traversal
  return thrust::system::detail::generic::detail::distance(
    first, last, typename thrust::iterator_traversal<InputIterator>::type());
} // end advance()

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
