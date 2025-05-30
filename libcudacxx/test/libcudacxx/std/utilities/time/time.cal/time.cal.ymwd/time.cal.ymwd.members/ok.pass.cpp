//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_weekday;

// constexpr bool ok() const noexcept;
//  Returns: m_.ok() && y_.ok().

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year               = cuda::std::chrono::year;
  using month              = cuda::std::chrono::month;
  using weekday            = cuda::std::chrono::weekday;
  using weekday_indexed    = cuda::std::chrono::weekday_indexed;
  using year_month_weekday = cuda::std::chrono::year_month_weekday;

  constexpr month January   = cuda::std::chrono::January;
  constexpr weekday Tuesday = cuda::std::chrono::Tuesday;

  static_assert(noexcept(cuda::std::declval<const year_month_weekday>().ok()));
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::std::declval<const year_month_weekday>().ok())>);

  static_assert(!year_month_weekday{}.ok(), "");

  static_assert(!year_month_weekday{year{-32768}, month{}, weekday_indexed{}}.ok(), ""); // All three bad

  static_assert(!year_month_weekday{year{-32768}, January, weekday_indexed{Tuesday, 1}}.ok(), ""); // Bad year
  static_assert(!year_month_weekday{year{2019}, month{}, weekday_indexed{Tuesday, 1}}.ok(), ""); // Bad month
  static_assert(!year_month_weekday{year{2019}, January, weekday_indexed{}}.ok(), ""); // Bad day

  static_assert(!year_month_weekday{year{-32768}, month{}, weekday_indexed{Tuesday, 1}}.ok(), ""); // Bad year & month
  static_assert(!year_month_weekday{year{2019}, month{}, weekday_indexed{}}.ok(), ""); // Bad month & day
  static_assert(!year_month_weekday{year{-32768}, January, weekday_indexed{}}.ok(), ""); // Bad year & day

  static_assert(year_month_weekday{year{2019}, January, weekday_indexed{Tuesday, 1}}.ok(), ""); // All OK

  for (unsigned i = 0; i <= 50; ++i)
  {
    year_month_weekday ym{year{2019}, January, weekday_indexed{Tuesday, i}};
    assert((ym.ok() == weekday_indexed{Tuesday, i}.ok()));
  }

  for (unsigned i = 0; i <= 50; ++i)
  {
    year_month_weekday ym{year{2019}, January, weekday_indexed{weekday{i}, 1}};
    assert((ym.ok() == weekday_indexed{weekday{i}, 1}.ok()));
  }

  for (unsigned i = 0; i <= 50; ++i)
  {
    year_month_weekday ym{year{2019}, month{i}, weekday_indexed{Tuesday, 1}};
    assert((ym.ok() == month{i}.ok()));
  }

  const int ymax = static_cast<int>(year::max());
  for (int i = ymax - 100; i <= ymax + 100; ++i)
  {
    year_month_weekday ym{year{i}, January, weekday_indexed{Tuesday, 1}};
    assert((ym.ok() == year{i}.ok()));
  }

  return 0;
}
