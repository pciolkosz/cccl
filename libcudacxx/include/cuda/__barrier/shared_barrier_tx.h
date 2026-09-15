//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_SHARED_BARRIER_TX_H
#define _CUDA___BARRIER_SHARED_BARRIER_TX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/shared_barrier.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 4)

_CCCL_BEGIN_NAMESPACE_CUDA

_CCCL_HOST_DEVICE_API inline void shared_barrier::expect_tx(::cuda::std::ptrdiff_t __transaction_count_update)
{
  _CCCL_ASSERT(0 <= __transaction_count_update, "Transaction count update must be non-negative.");
  _CCCL_ASSERT(__transaction_count_update <= __max_transaction_count_update(),
               "Transaction count update cannot exceed the mbarrier transaction count limit.");

  NV_IF_TARGET(NV_PROVIDES_SM_90, (__expect_tx(__transaction_count_update); return;))

  __unsupported_storage();
}

[[nodiscard]] _CCCL_HOST_DEVICE_API inline shared_barrier::arrival_token shared_barrier::arrive_tx(
  ::cuda::std::ptrdiff_t __arrive_count_update, ::cuda::std::ptrdiff_t __transaction_count_update)
{
  _CCCL_ASSERT(1 <= __arrive_count_update, "Arrival count update must be at least one.");
  _CCCL_ASSERT(__arrive_count_update <= shared_barrier::max(),
               "Arrival count update cannot exceed shared_barrier::max().");
  _CCCL_ASSERT(0 <= __transaction_count_update, "Transaction count update must be non-negative.");
  _CCCL_ASSERT(__transaction_count_update <= __max_transaction_count_update(),
               "Transaction count update cannot exceed the mbarrier transaction count limit.");

  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (return arrival_token(__arrive_tx(__arrive_count_update, __transaction_count_update));))

  __unsupported_storage();
}

_CCCL_END_NAMESPACE_CUDA

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 4)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BARRIER_SHARED_BARRIER_TX_H
