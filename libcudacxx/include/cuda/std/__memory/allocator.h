// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_ALLOCATOR_H
#define _LIBCUDACXX___MEMORY_ALLOCATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/allocator.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/allocate_at_least.h>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__new_>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstddef>

#if defined(_CCCL_HAS_CONSTEXPR_ALLOCATION) && !_CCCL_COMPILER(NVRTC)
#  include <memory>
#endif // _CCCL_HAS_CONSTEXPR_ALLOCATION && !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER <= 2017
// These specializations shouldn't be marked _LIBCUDACXX_DEPRECATED.
// Specializing allocator<void> is deprecated, but not using it.
template <>
class _CCCL_TYPE_VISIBILITY_DEFAULT allocator<void>
{
public:
  using pointer _LIBCUDACXX_DEPRECATED       = void*;
  using const_pointer _LIBCUDACXX_DEPRECATED = const void*;
  using value_type _LIBCUDACXX_DEPRECATED    = void;

  template <class _Up>
  struct _LIBCUDACXX_DEPRECATED rebind
  {
    using other = allocator<_Up>;
  };
};

template <>
class _CCCL_TYPE_VISIBILITY_DEFAULT allocator<const void>
{
public:
  using pointer _LIBCUDACXX_DEPRECATED       = const void*;
  using const_pointer _LIBCUDACXX_DEPRECATED = const void*;
  using value_type _LIBCUDACXX_DEPRECATED    = const void;

  template <class _Up>
  struct _LIBCUDACXX_DEPRECATED rebind
  {
    using other = allocator<_Up>;
  };
};
#endif // _CCCL_STD_VER <= 2017

// This class provides a non-trivial default constructor to the class that derives from it
// if the condition is satisfied.
//
// The second template parameter exists to allow giving a unique type to __non_trivial_if,
// which makes it possible to avoid breaking the ABI when making this a base class of an
// existing class. Without that, imagine we have classes D1 and D2, both of which used to
// have no base classes, but which now derive from __non_trivial_if. The layout of a class
// that inherits from both D1 and D2 will change because the two __non_trivial_if base
// classes are not allowed to share the same address.
//
// By making those __non_trivial_if base classes unique, we work around this problem and
// it is safe to start deriving from __non_trivial_if in existing classes.
template <bool _Cond, class _Unique>
struct __non_trivial_if
{};

template <class _Unique>
struct __non_trivial_if<true, _Unique>
{
  _CCCL_API constexpr __non_trivial_if() noexcept {}
};

// allocator
//
// Note: For ABI compatibility between C++20 and previous standards, we make
//       allocator<void> trivial in C++20.

template <class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT allocator : private __non_trivial_if<!_CCCL_TRAIT(is_void, _Tp), allocator<_Tp>>
{
  static_assert(!_CCCL_TRAIT(is_volatile, _Tp), "std::allocator does not support volatile types");

public:
  using size_type                              = size_t;
  using difference_type                        = ptrdiff_t;
  using value_type                             = _Tp;
  using propagate_on_container_move_assignment = true_type;
  using is_always_equal                        = true_type;

  _CCCL_CONSTEXPR_CXX20 allocator() noexcept = default;

  template <class _Up>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 allocator(const allocator<_Up>&) noexcept
  {}

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API inline _CCCL_CONSTEXPR_CXX20_ALLOCATION _Tp* allocate(size_t __n)
  {
    if (__n > allocator_traits<allocator>::max_size(*this))
    {
      __throw_bad_array_new_length();
    }
#if defined(_CCCL_HAS_CONSTEXPR_ALLOCATION)
    if (_CUDA_VSTD::is_constant_evaluated())
    {
      return ::std::allocator<_Tp>{}.allocate(__n);
    }
#endif // _CCCL_HAS_CONSTEXPR_ALLOCATION
    {
      return static_cast<_Tp*>(_CUDA_VSTD::__cccl_allocate(__n * sizeof(_Tp), alignof(_Tp)));
    }
  }

#if _CCCL_STD_VER >= 2023
  [[nodiscard]] _CCCL_API constexpr allocation_result<_Tp*> allocate_at_least(size_t __n)
  {
    return {allocate(__n), __n};
  }
#endif // _CCCL_HAS_CONSTEXPR_ALLOCATION

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20_ALLOCATION void deallocate(_Tp* __p, size_t __n) noexcept
  {
#if defined(_CCCL_HAS_CONSTEXPR_ALLOCATION)
    if (_CUDA_VSTD::is_constant_evaluated())
    {
      return ::std::allocator<_Tp>{}.deallocate(__p, __n);
    }
    else
#endif // _CCCL_STD_VER >= 2020
    {
      _CUDA_VSTD::__cccl_deallocate((void*) __p, __n * sizeof(_Tp), alignof(_Tp));
    }
  }

  // C++20 Removed members
#if _CCCL_STD_VER <= 2017
  using pointer _LIBCUDACXX_DEPRECATED         = _Tp*;
  using const_pointer _LIBCUDACXX_DEPRECATED   = const _Tp*;
  using reference _LIBCUDACXX_DEPRECATED       = _Tp&;
  using const_reference _LIBCUDACXX_DEPRECATED = const _Tp&;

  template <class _Up>
  struct _LIBCUDACXX_DEPRECATED rebind
  {
    using other = allocator<_Up>;
  };

  _LIBCUDACXX_DEPRECATED _CCCL_API inline pointer address(reference __x) const noexcept
  {
    return _CUDA_VSTD::addressof(__x);
  }
  _LIBCUDACXX_DEPRECATED _CCCL_API inline const_pointer address(const_reference __x) const noexcept
  {
    return _CUDA_VSTD::addressof(__x);
  }

  [[nodiscard]] _CCCL_API inline _LIBCUDACXX_DEPRECATED _Tp* allocate(size_t __n, const void*)
  {
    return allocate(__n);
  }

  _LIBCUDACXX_DEPRECATED _CCCL_API inline size_type max_size() const noexcept
  {
    return size_type(~0) / sizeof(_Tp);
  }

  template <class _Up, class... _Args>
  _LIBCUDACXX_DEPRECATED _CCCL_API inline void construct(_Up* __p, _Args&&... __args)
  {
    ::new ((void*) __p) _Up(_CUDA_VSTD::forward<_Args>(__args)...);
  }

  _LIBCUDACXX_DEPRECATED _CCCL_API inline void destroy(pointer __p) noexcept
  {
    __p->~_Tp();
  }
#endif // _CCCL_STD_VER <= 2017
};

template <class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT
allocator<const _Tp> : private __non_trivial_if<!_CCCL_TRAIT(is_void, _Tp), allocator<const _Tp>>
{
  static_assert(!_CCCL_TRAIT(is_volatile, _Tp), "std::allocator does not support volatile types");

public:
  using size_type                              = size_t;
  using difference_type                        = ptrdiff_t;
  using value_type                             = const _Tp;
  using propagate_on_container_move_assignment = true_type;
  using is_always_equal                        = true_type;

  _CCCL_CONSTEXPR_CXX20 allocator() noexcept = default;

  template <class _Up>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 allocator(const allocator<_Up>&) noexcept
  {}

  [[nodiscard]] _CCCL_API inline _CCCL_CONSTEXPR_CXX20 const _Tp* allocate(size_t __n)
  {
    if (__n > allocator_traits<allocator>::max_size(*this))
    {
      __throw_bad_array_new_length();
    }
    if (_CUDA_VSTD::is_constant_evaluated())
    {
      return static_cast<const _Tp*>(::operator new(__n * sizeof(_Tp)));
    }
    else
    {
      return static_cast<const _Tp*>(_CUDA_VSTD::__cccl_allocate(__n * sizeof(_Tp), alignof(_Tp)));
    }
  }

#if _CCCL_STD_VER >= 2023
  [[nodiscard]] _CCCL_API constexpr allocation_result<const _Tp*> allocate_at_least(size_t __n)
  {
    return {allocate(__n), __n};
  }
#endif // _CCCL_STD_VER >= 2023

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void deallocate(const _Tp* __p, size_t __n) noexcept
  {
    if (_CUDA_VSTD::is_constant_evaluated())
    {
      ::operator delete(const_cast<_Tp*>(__p));
    }
    else
    {
      _CUDA_VSTD::__cccl_deallocate((void*) const_cast<_Tp*>(__p), __n * sizeof(_Tp), alignof(_Tp));
    }
  }

  // C++20 Removed members
#if _CCCL_STD_VER <= 2017
  using pointer _LIBCUDACXX_DEPRECATED         = const _Tp*;
  using const_pointer _LIBCUDACXX_DEPRECATED   = const _Tp*;
  using reference _LIBCUDACXX_DEPRECATED       = const _Tp&;
  using const_reference _LIBCUDACXX_DEPRECATED = const _Tp&;

  template <class _Up>
  struct _LIBCUDACXX_DEPRECATED rebind
  {
    using other = allocator<_Up>;
  };

  _LIBCUDACXX_DEPRECATED _CCCL_API inline const_pointer address(const_reference __x) const noexcept
  {
    return _CUDA_VSTD::addressof(__x);
  }

  [[nodiscard]] _CCCL_API inline _LIBCUDACXX_DEPRECATED const _Tp* allocate(size_t __n, const void*)
  {
    return allocate(__n);
  }

  _LIBCUDACXX_DEPRECATED _CCCL_API inline size_type max_size() const noexcept
  {
    return size_type(~0) / sizeof(_Tp);
  }

  template <class _Up, class... _Args>
  _LIBCUDACXX_DEPRECATED _CCCL_API inline void construct(_Up* __p, _Args&&... __args)
  {
    ::new ((void*) __p) _Up(_CUDA_VSTD::forward<_Args>(__args)...);
  }

  _LIBCUDACXX_DEPRECATED _CCCL_API inline void destroy(pointer __p) noexcept
  {
    __p->~_Tp();
  }
#endif // _CCCL_STD_VER <= 2017
};

template <class _Tp, class _Up>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator==(const allocator<_Tp>&, const allocator<_Up>&) noexcept
{
  return true;
}

template <class _Tp, class _Up>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator!=(const allocator<_Tp>&, const allocator<_Up>&) noexcept
{
  return false;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_ALLOCATOR_H
