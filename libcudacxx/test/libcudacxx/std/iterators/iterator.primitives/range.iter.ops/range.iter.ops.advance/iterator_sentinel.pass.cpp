//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// ranges::advance(it, sent)

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <bool Count, class It>
__host__ __device__ constexpr void check_assignable(int* first, int* last, int* expected)
{
  {
    It it(first);
    auto sent = assignable_sentinel(It(last));
    cuda::std::ranges::advance(it, sent);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::advance(it, sent)), void>);
    assert(base(it) == expected);
  }

  // Count operations
  if constexpr (Count)
  {
    auto it   = stride_counting_iterator(It(first));
    auto sent = assignable_sentinel(stride_counting_iterator(It(last)));
    cuda::std::ranges::advance(it, sent);
    assert(base(base(it)) == expected);
    assert(it.stride_count() == 0); // because we got here by assigning from last, not by incrementing
  }
}

template <bool Count, class It>
__host__ __device__ constexpr void check_sized_sentinel(int* first, int* last, int* expected)
{
  auto size = (last - first);

  {
    It it(first);
    auto sent = distance_apriori_sentinel(size);
    cuda::std::ranges::advance(it, sent);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::advance(it, sent)), void>);
    assert(base(it) == expected);
  }

  // Count operations
  if constexpr (Count)
  {
    auto it   = stride_counting_iterator(It(first));
    auto sent = distance_apriori_sentinel(size);
    cuda::std::ranges::advance(it, sent);
    if constexpr (cuda::std::random_access_iterator<It>)
    {
      assert(it.stride_count() == 1);
    }
    else
    {
      assert(it.stride_count() == size);
    }
  }
}

template <bool Count, class It>
__host__ __device__ constexpr void check_sentinel(int* first, int* last, int* expected)
{
  auto size = (last - first);
  unused(size);

  {
    It it(first);
    auto sent = sentinel_wrapper(It(last));
    cuda::std::ranges::advance(it, sent);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::advance(it, sent)), void>);
    assert(base(it) == expected);
  }

  // Count operations
  if constexpr (Count)
  {
    auto it   = stride_counting_iterator(It(first));
    auto sent = sentinel_wrapper(stride_counting_iterator(It(last)));
    cuda::std::ranges::advance(it, sent);
    assert(it.stride_count() == size);
  }
}

__host__ __device__ constexpr bool test()
{
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (int n = 0; n != 10; ++n)
  {
    check_assignable<false, cpp17_input_iterator<int*>>(range, range + n, range + n);
    check_assignable<false, cpp20_input_iterator<int*>>(range, range + n, range + n);
    check_assignable<true, forward_iterator<int*>>(range, range + n, range + n);
    check_assignable<true, bidirectional_iterator<int*>>(range, range + n, range + n);
    check_assignable<true, random_access_iterator<int*>>(range, range + n, range + n);
    check_assignable<true, contiguous_iterator<int*>>(range, range + n, range + n);
    check_assignable<true, int*>(range, range + n, range + n);

    check_sized_sentinel<false, cpp17_input_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<false, cpp20_input_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<true, forward_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<true, bidirectional_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<true, random_access_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<true, contiguous_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<true, int*>(range, range + n, range + n);

    check_sentinel<false, cpp17_input_iterator<int*>>(range, range + n, range + n);
    check_sentinel<false, cpp20_input_iterator<int*>>(range, range + n, range + n);
    check_sentinel<true, forward_iterator<int*>>(range, range + n, range + n);
    check_sentinel<true, bidirectional_iterator<int*>>(range, range + n, range + n);
    check_sentinel<true, random_access_iterator<int*>>(range, range + n, range + n);
    check_sentinel<true, contiguous_iterator<int*>>(range, range + n, range + n);
    check_sentinel<true, int*>(range, range + n, range + n);
  }

  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
