//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// member_object_pointer

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_member_object_pointer_imp()
{
  static_assert(!cuda::std::is_void<T>::value, "");
  static_assert(!cuda::std::is_null_pointer<T>::value, "");
  static_assert(!cuda::std::is_integral<T>::value, "");
  static_assert(!cuda::std::is_floating_point<T>::value, "");
  static_assert(!cuda::std::is_array<T>::value, "");
  static_assert(!cuda::std::is_pointer<T>::value, "");
  static_assert(!cuda::std::is_lvalue_reference<T>::value, "");
  static_assert(!cuda::std::is_rvalue_reference<T>::value, "");
  static_assert(cuda::std::is_member_object_pointer<T>::value, "");
  static_assert(!cuda::std::is_member_function_pointer<T>::value, "");
  static_assert(!cuda::std::is_enum<T>::value, "");
  static_assert(!cuda::std::is_union<T>::value, "");
  static_assert(!cuda::std::is_class<T>::value, "");
  static_assert(!cuda::std::is_function<T>::value, "");
}

template <class T>
__host__ __device__ void test_member_object_pointer()
{
  test_member_object_pointer_imp<T>();
  test_member_object_pointer_imp<const T>();
  test_member_object_pointer_imp<volatile T>();
  test_member_object_pointer_imp<const volatile T>();
}

class Class
{};

struct incomplete_type;

int main(int, char**)
{
  test_member_object_pointer<int Class::*>();

  //  LWG#2582
  static_assert(!cuda::std::is_member_object_pointer<incomplete_type>::value, "");

  return 0;
}
