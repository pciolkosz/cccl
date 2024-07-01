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

#include "cpos.hpp"
#include "meta.hpp"

// Must be the last include
#include "prologue.hpp"

namespace USTDEX_NAMESPACE
{

template <class Rcvr>
constexpr Rcvr* _rcvr_ref(Rcvr& rcvr) noexcept
{
  return &rcvr;
}

template <class Rcvr>
constexpr Rcvr* _rcvr_ref(Rcvr* rcvr) noexcept
{
  return rcvr;
}

template <class Rcvr>
using _rcvr_ref_t = decltype(ustdex::_rcvr_ref(DECLVAL(Rcvr)));

} // namespace USTDEX_NAMESPACE

#include "epilogue.hpp"
