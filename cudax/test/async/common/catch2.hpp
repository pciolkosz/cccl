//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cuda/experimental/async.cuh"

#if !USTDEX_HOST_ONLY()
// disable exceptions in Catch2 when compiling for device:
#  define CATCH_CONFIG_DISABLE_EXCEPTIONS
#endif

#include <catch2/catch.hpp>

// In device code, a few of the Catch2 macros are not supported.
#if !USTDEX_HOST_ONLY()
#  undef CHECK
#  define CHECK(expr)                                                         \
    if (expr)                                                                 \
    {                                                                         \
    }                                                                         \
    else                                                                      \
    {                                                                         \
      printf("Test failed: \"%s\", %s(%d)", #expr, __FILE__, (int) __LINE__); \
      asm("trap;");                                                           \
    }

#  undef FAIL
#  define FAIL(msg)                                                         \
    {                                                                       \
      printf("Test failed: \"%s\", %s(%d)", msg, __FILE__, (int) __LINE__); \
      asm("trap;");                                                         \
    }

#  undef CHECK_FALSE
#  define CHECK_FALSE(expr) CHECK(!(expr))

#  undef REQUIRE
#  define REQUIRE(expr) CHECK(expr)
#endif

namespace cuda::experimental
{
}

namespace cudax = cuda::experimental;
