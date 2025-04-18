//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month_weekday;

// constexpr chrono::month month() const noexcept;
//  Returns: wd_

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month_weekday   = cuda::std::chrono::month_weekday;
  using month           = cuda::std::chrono::month;
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  constexpr weekday Sunday = cuda::std::chrono::Sunday;

  static_assert(noexcept(cuda::std::declval<const month_weekday>().month()));
  static_assert(cuda::std::is_same_v<month, decltype(cuda::std::declval<const month_weekday>().month())>);

  static_assert(month_weekday{}.month() == month{}, "");

  for (unsigned i = 1; i <= 50; ++i)
  {
    month_weekday md(month{i}, weekday_indexed{Sunday, 1});
    assert(static_cast<unsigned>(md.month()) == i);
  }

  return 0;
}
