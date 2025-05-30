//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_weekday;

//  year_month_weekday() = default;
//  constexpr year_month_weekday(const chrono::year& y, const chrono::month& m,
//                               const chrono::weekday_indexed& wdi) noexcept;
//
//  Effects:  Constructs an object of type year_month_weekday by initializing
//                y_ with y, m_ with m, and wdi_ with wdi.
//
//  constexpr chrono::year                       year() const noexcept;
//  constexpr chrono::month                     month() const noexcept;
//  constexpr chrono::weekday                 weekday() const noexcept;
//  constexpr unsigned                          index() const noexcept;
//  constexpr chrono::weekday_indexed weekday_indexed() const noexcept;
//  constexpr bool                                 ok() const noexcept;

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

  static_assert(noexcept(year_month_weekday{}));
  static_assert(noexcept(year_month_weekday{year{1}, month{1}, weekday_indexed{Tuesday, 1}}));

  constexpr year_month_weekday ym0{};
  static_assert(ym0.year() == year{}, "");
  static_assert(ym0.month() == month{}, "");
  static_assert(ym0.weekday() == weekday{}, "");
  static_assert(ym0.index() == 0, "");
  static_assert(ym0.weekday_indexed() == weekday_indexed{}, "");
  static_assert(!ym0.ok(), "");

  constexpr year_month_weekday ym1{year{2019}, January, weekday_indexed{Tuesday, 1}};
  static_assert(ym1.year() == year{2019}, "");
  static_assert(ym1.month() == January, "");
  static_assert(ym1.weekday() == Tuesday, "");
  static_assert(ym1.index() == 1, "");
  static_assert(ym1.weekday_indexed() == weekday_indexed{Tuesday, 1}, "");
  static_assert(ym1.ok(), "");

  return 0;
}
