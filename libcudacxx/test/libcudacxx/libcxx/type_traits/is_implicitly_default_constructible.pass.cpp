//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: gcc-5
// Before GCC 6, this trait fails. See https://stackoverflow.com/q/41799015/627587.

// <cuda/std/type_traits>

// __is_implicitly_default_constructible<Tp>

#include <cuda/std/type_traits>

#include "test_macros.h"

struct ExplicitlyDefaultConstructible1
{
  explicit ExplicitlyDefaultConstructible1() = default;
};

struct ExplicitlyDefaultConstructible2
{
  __host__ __device__ explicit ExplicitlyDefaultConstructible2() {}
};

struct ImplicitlyDefaultConstructible1
{
  __host__ __device__ ImplicitlyDefaultConstructible1() {}
};

struct ImplicitlyDefaultConstructible2
{
  ImplicitlyDefaultConstructible2() = default;
};

struct NonDefaultConstructible1
{
  NonDefaultConstructible1() = delete;
};

struct NonDefaultConstructible2
{
  explicit NonDefaultConstructible2() = delete;
};

struct NonDefaultConstructible3
{
  __host__ __device__ NonDefaultConstructible3(NonDefaultConstructible3&&) {}
};

struct ProtectedDefaultConstructible
{
protected:
  ProtectedDefaultConstructible() = default;
};

struct PrivateDefaultConstructible
{
private:
  PrivateDefaultConstructible() = default;
};

struct Base
{};

struct ProtectedDefaultConstructibleWithBase : Base
{
protected:
  ProtectedDefaultConstructibleWithBase() = default;
};

struct PrivateDefaultConstructibleWithBase : Base
{
private:
  PrivateDefaultConstructibleWithBase() = default;
};

static_assert(!cuda::std::__is_implicitly_default_constructible<ExplicitlyDefaultConstructible1>::value, "");
static_assert(!cuda::std::__is_implicitly_default_constructible<ExplicitlyDefaultConstructible2>::value, "");
static_assert(cuda::std::__is_implicitly_default_constructible<ImplicitlyDefaultConstructible1>::value, "");
static_assert(cuda::std::__is_implicitly_default_constructible<ImplicitlyDefaultConstructible2>::value, "");
static_assert(!cuda::std::__is_implicitly_default_constructible<NonDefaultConstructible1>::value, "");
static_assert(!cuda::std::__is_implicitly_default_constructible<NonDefaultConstructible2>::value, "");
static_assert(!cuda::std::__is_implicitly_default_constructible<NonDefaultConstructible3>::value, "");
static_assert(!cuda::std::__is_implicitly_default_constructible<ProtectedDefaultConstructible>::value, "");
static_assert(!cuda::std::__is_implicitly_default_constructible<PrivateDefaultConstructible>::value, "");
#if !TEST_COMPILER(GCC, <, 8) // GCC 6 + 7 complain about implicit conversion
static_assert(!cuda::std::__is_implicitly_default_constructible<ProtectedDefaultConstructibleWithBase>::value, "");
static_assert(!cuda::std::__is_implicitly_default_constructible<PrivateDefaultConstructibleWithBase>::value, "");
#endif // !TEST_COMPILER(GCC, <, 8)

int main(int, char**)
{
  return 0;
}
