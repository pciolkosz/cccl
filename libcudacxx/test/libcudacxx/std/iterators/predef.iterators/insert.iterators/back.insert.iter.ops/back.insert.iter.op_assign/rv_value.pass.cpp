//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// back_insert_iterator

// requires CopyConstructible<Cont::value_type>
//   back_insert_iterator<Cont>&
//   operator=(Cont::value_type&& value);

#include <cuda/std/cassert>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>

#include "MoveOnly.h"
#include "test_macros.h"

template <class C>
__host__ __device__ void test(C c)
{
  cuda::std::back_insert_iterator<C> i(c);
  i = typename C::value_type();
  assert(c.back() == typename C::value_type());
}

int main(int, char**)
{
  test(cuda::std::inplace_vector<MoveOnly, 3>());

  return 0;
}
