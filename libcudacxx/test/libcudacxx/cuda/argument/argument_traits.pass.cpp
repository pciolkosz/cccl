//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/argument>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_macros.h"

enum class color
{
  red,
  green,
  blue
};

TEST_FUNC void test()
{
  // --- __is_single_value_v on plain types ---

  // Arithmetic types are single values
  static_assert(cuda::__is_single_value_v<int>);
  static_assert(cuda::__is_single_value_v<float>);
  static_assert(cuda::__is_single_value_v<double>);
  static_assert(cuda::__is_single_value_v<const int>);

  // Enums are single values
  static_assert(cuda::__is_single_value_v<color>);

  // span<T, 1> (span<T, 1>) is a single value
  static_assert(cuda::__is_single_value_v<cuda::std::span<int, 1>>);

  // Pointers are not single values
  static_assert(!cuda::__is_single_value_v<int*>);

  // Spans are not single values (except extent 1)
  static_assert(!cuda::__is_single_value_v<cuda::std::span<int>>);
  static_assert(!cuda::__is_single_value_v<cuda::std::span<int, 4>>);

  // Arrays are not single values
  static_assert(!cuda::__is_single_value_v<cuda::std::array<int, 3>>);

  // --- argument_traits: is_deferred ---

  static_assert(!cuda::argument_traits<int>::is_deferred);
  static_assert(!cuda::argument_traits<cuda::dynamic_argument<int>>::is_deferred);
  static_assert(!cuda::argument_traits<cuda::static_argument<42>>::is_deferred);
  static_assert(cuda::argument_traits<cuda::deferred_argument<cuda::std::span<int, 1>>>::is_deferred);
  static_assert(cuda::argument_traits<cuda::deferred_argument<cuda::std::span<int>>>::is_deferred);

  // --- argument_traits: value_type ---

  static_assert(cuda::std::is_same_v<cuda::argument_traits<int>::value_type, int>);
  static_assert(cuda::std::is_same_v<cuda::argument_traits<cuda::dynamic_argument<int>>::value_type, int>);
  static_assert(cuda::std::is_same_v<cuda::argument_traits<cuda::static_argument<42>>::value_type, int>);
  static_assert(
    cuda::std::is_same_v<cuda::argument_traits<cuda::deferred_argument<cuda::std::span<int, 1>>>::value_type,
                         cuda::std::span<int, 1>>);
  static_assert(
    cuda::std::is_same_v<cuda::argument_traits<cuda::deferred_argument<cuda::std::span<int>>>::value_type,
                         cuda::std::span<int>>);

  // --- __is_single_value_v on unwrapped wrapper types ---

  // dynamic_argument<int> unwraps to int → single value
  static_assert(cuda::__is_single_value_v<cuda::argument_traits<cuda::dynamic_argument<int>>::value_type>);

  // dynamic_argument<span<int>> unwraps to span<int> → not single value
  static_assert(!cuda::__is_single_value_v<cuda::argument_traits<cuda::dynamic_argument<cuda::std::span<int>>>::value_type>);

  // dynamic_argument<int*> unwraps to int* → not single value
  static_assert(!cuda::__is_single_value_v<cuda::argument_traits<cuda::dynamic_argument<int*>>::value_type>);

  // static_argument<42> → int → single value
  static_assert(cuda::__is_single_value_v<cuda::argument_traits<cuda::static_argument<42>>::value_type>);

  // static_argument with array → not single value
  using arr_t = cuda::std::array<int, 3>;
  static_assert(!cuda::__is_single_value_v<cuda::argument_traits<cuda::static_argument<arr_t{1, 2, 3}>>::value_type>);

  // --- Free function bounds on plain values ---
  static_assert(cuda::argument_static_min(42) == cuda::std::numeric_limits<int>::lowest());
  static_assert(cuda::argument_static_max(42) == cuda::std::numeric_limits<int>::max());
  static_assert(cuda::argument_min(42) == cuda::std::numeric_limits<int>::lowest());
  static_assert(cuda::argument_max(42) == cuda::std::numeric_limits<int>::max());
}

int main(int, char**)
{
  test();
  return 0;
}
