//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// template<borrowed_range R>
//   requires convertible-to-non-slicing<iterator_t<R>, I> &&
//            convertible_to<sentinel_t<R>, S>
// constexpr subrange(R&& r, make-unsigned-like-t<iter_difference_t<I>> n)
//   requires (K == subrange_kind::sized);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

struct BorrowedRange
{
  __host__ __device__ constexpr explicit BorrowedRange(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr int* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return end_;
  }

private:
  int* begin_;
  int* end_;
};

template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<::BorrowedRange> = true;

__host__ __device__ constexpr bool test()
{
  int buff[]     = {1, 2, 3, 4, 5, 6, 7, 8};
  using Subrange = cuda::std::ranges::subrange<int*, int*, cuda::std::ranges::subrange_kind::sized>;

  // Test with an empty range
  {
    BorrowedRange range(buff, buff);
    Subrange subrange(range, 0);
    assert(subrange.size() == 0);
  }

  // Test with non-empty ranges
  {
    BorrowedRange range(buff, buff + 1);
    Subrange subrange(range, 1);
    assert(subrange.size() == 1);
  }
  {
    BorrowedRange range(buff, buff + 2);
    Subrange subrange(range, 2);
    unused(subrange);
    assert(subrange[0] == 1);
    assert(subrange[1] == 2);
    assert(subrange.size() == 2);
  }
  {
    BorrowedRange range(buff, buff + 8);
    Subrange subrange(range, 8);
    assert(subrange[0] == 1);
    assert(subrange[1] == 2);
    // ...
    assert(subrange[7] == 8);
    assert(subrange.size() == 8);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
