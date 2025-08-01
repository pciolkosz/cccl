//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_ADDRESS_SPACE_H
#define _CUDA___MEMORY_ADDRESS_SPACE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__utility/to_underlying.h>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

enum class address_space
{
  global,
  shared,
  constant,
  local,
  grid_constant,
  cluster_shared,
  __max,
};

[[nodiscard]] _CCCL_DEVICE_API constexpr bool __cccl_is_valid_address_space(address_space __space) noexcept
{
  const auto __v = _CUDA_VSTD::to_underlying(__space);
  return __v >= 0 && __v < _CUDA_VSTD::to_underlying(address_space::__max);
}

[[nodiscard]] _CCCL_DEVICE_API inline bool is_address_from(const void* __ptr, address_space __space)
{
  _CCCL_ASSERT(__ptr != nullptr, "invalid pointer");
  _CCCL_ASSERT(_CUDA_DEVICE::__cccl_is_valid_address_space(__space), "invalid address space");

  switch (__space)
  {
    case address_space::global:
      return static_cast<bool>(::__isGlobal(__ptr));
    case address_space::shared:
      return static_cast<bool>(::__isShared(__ptr));
    case address_space::constant:
      return static_cast<bool>(::__isConstant(__ptr));
    case address_space::local:
      return static_cast<bool>(::__isLocal(__ptr));
    case address_space::grid_constant:
#  if _CCCL_HAS_GRID_CONSTANT()
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70, (return static_cast<bool>(::__isGridConstant(__ptr));), (return false;))
#  else // ^^^ _CCCL_HAS_GRID_CONSTANT() ^^^ / vvv !_CCCL_HAS_GRID_CONSTANT() vvv
      return false;
#  endif // ^^^ !_CCCL_HAS_GRID_CONSTANT() ^^^
    case address_space::cluster_shared:
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return static_cast<bool>(::__isClusterShared(__ptr));), (return false;))
    default:
      return false;
  }
}

template <typename T>
[[nodiscard]] _CCCL_DEVICE_API inline bool is_object_from(T& __obj, address_space __space)
{
  return _CUDA_DEVICE::is_address_from(_CUDA_VSTD::addressof(__obj), __space);
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA___MEMORY_ADDRESS_SPACE_H
