//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr basic_string_view(nullptr_t) = delete;

#include <cuda/std/cstddef>
#include <cuda/std/string_view>

template <class SV>
__host__ __device__ constexpr void test_nullptr_constructor()
{
  [[maybe_unused]] SV sv{nullptr};
}

__host__ __device__ constexpr bool test()
{
  test_nullptr_constructor<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_nullptr_constructor<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_nullptr_constructor<cuda::std::u16string_view>();
  test_nullptr_constructor<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_nullptr_constructor<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
