//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>

#include <cuda/experimental/memory_resource.cuh>

#include "test_resource.cuh"
#include <testing.cuh>

// TODO we need more async_resource_ref  and resource_ref tests
C2H_TEST("async_resource_ref assignment", "[memory_resource]")
{
  dummy_test_resource_n<42> res_42;
  dummy_test_resource_n<43> res_43;

  cudax::async_resource_ref<cudax::host_accessible> ref{res_42};
  CHECK(ref.allocate(100, 1) == reinterpret_cast<void*>(42));

  ref = cudax::async_resource_ref<cudax::host_accessible>{res_43};
  CHECK(ref.allocate(100, 1) == reinterpret_cast<void*>(43));

  // Check that we can copy an async_resource_ref to an any_async_resource after assignment
  cudax::any_async_resource<cudax::host_accessible> any_ref{ref};
  CHECK(any_ref.allocate(100, 1) == reinterpret_cast<void*>(43));
}

C2H_TEST("resource_ref assignment", "[memory_resource]")
{
  dummy_test_resource_n<42> res_42;
  dummy_test_resource_n<43> res_43;

  cudax::resource_ref<cudax::host_accessible> ref{res_42};
  CHECK(ref.allocate(100, 1) == reinterpret_cast<void*>(42));

  ref = cudax::resource_ref<cudax::host_accessible>{res_43};
  CHECK(ref.allocate(100, 1) == reinterpret_cast<void*>(43));

  // Check that we can copy a resource_ref to an any_resource after assignment
  cudax::any_resource<cudax::host_accessible> any_ref{ref};
  CHECK(any_ref.allocate(100, 1) == reinterpret_cast<void*>(43));
}
