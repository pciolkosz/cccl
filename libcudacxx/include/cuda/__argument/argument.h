//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ARGUMENT_ARGUMENT_H
#define _CUDA___ARGUMENT_ARGUMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__argument/argument_bounds.h>
#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// =====================================================================
// __argument_element_type — extract element type from arg type
// =====================================================================

template <class _Tp>
struct __argument_element_type
{
  using type = _Tp;
};

template <class _Tp, size_t _Extent>
struct __argument_element_type<::cuda::std::span<_Tp, _Extent>>
{
  using type = _Tp;
};

template <class _Tp, size_t _N>
struct __argument_element_type<::cuda::std::array<_Tp, _N>>
{
  using type = _Tp;
};

template <class _Tp>
using __argument_element_type_t = typename __argument_element_type<_Tp>::type;

// =====================================================================
// static_argument
// =====================================================================

//! @brief Wraps a compile-time constant argument value.
template <auto _Value>
struct static_argument
{
  using value_type                  = decltype(_Value);
  using element_type                = __argument_element_type_t<value_type>;
  static constexpr value_type value = _Value;
};

// =====================================================================
// dynamic_argument
// =====================================================================

//! @brief Wraps a runtime argument value with optional bounds.
template <class _Arg, class _StaticBounds = __no_bounds>
struct dynamic_argument
{
  using element_type = __argument_element_type_t<_Arg>;

  _Arg arg;
  _CCCL_NO_UNIQUE_ADDRESS _StaticBounds __static_bounds_;
  runtime_argument_bounds<element_type> __runtime_bounds_;

private:
  _CCCL_API constexpr void __validate_element(element_type __val) noexcept
  {
    if constexpr (!::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
    {
      _CCCL_ASSERT(__val >= __static_bounds_.min, "dynamic_argument value is below static minimum bound");
      _CCCL_ASSERT(__val <= __static_bounds_.max, "dynamic_argument value is above static maximum bound");
    }
    _CCCL_ASSERT(__val >= __runtime_bounds_.min, "dynamic_argument value is below runtime minimum bound");
    _CCCL_ASSERT(__val <= __runtime_bounds_.max, "dynamic_argument value is above runtime maximum bound");
  }

  _CCCL_API constexpr void __validate() noexcept
  {
    if constexpr (::cuda::std::is_arithmetic_v<_Arg> || ::cuda::std::is_enum_v<_Arg>)
    {
      __validate_element(arg);
    }
    else if constexpr (::cuda::std::__is_cuda_std_span_v<_Arg>)
    {
      for (size_t __i = 0; __i < arg.size(); ++__i)
      {
        __validate_element(arg[__i]);
      }
    }
  }

public:
  _CCCL_API constexpr dynamic_argument(_Arg __arg) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{}
  {}

  template <auto _Min, auto _Max>
  _CCCL_API constexpr dynamic_argument(_Arg __arg, static_argument_bounds<_Min, _Max> __sb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{}
  {
    __validate();
  }

  template <class _BoundsTp>
  _CCCL_API constexpr dynamic_argument(_Arg __arg, runtime_argument_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{static_cast<element_type>(__rb.min), static_cast<element_type>(__rb.max)}
  {
    __validate();
  }

  template <auto _Min, auto _Max, class _BoundsTp>
  _CCCL_API constexpr dynamic_argument(
    _Arg __arg, static_argument_bounds<_Min, _Max> __sb, runtime_argument_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{static_cast<element_type>(__rb.min), static_cast<element_type>(__rb.max)}
  {
    __validate();
  }
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg>
_CCCL_HOST_DEVICE dynamic_argument(_Arg) -> dynamic_argument<_Arg>;

template <class _Arg, auto _Min, auto _Max>
_CCCL_HOST_DEVICE dynamic_argument(_Arg, static_argument_bounds<_Min, _Max>)
  -> dynamic_argument<_Arg, static_argument_bounds<_Min, _Max>>;

template <class _Arg, class _Tp>
_CCCL_HOST_DEVICE dynamic_argument(_Arg, runtime_argument_bounds<_Tp>) -> dynamic_argument<_Arg>;

template <class _Arg, auto _Min, auto _Max, class _Tp>
_CCCL_HOST_DEVICE dynamic_argument(_Arg, static_argument_bounds<_Min, _Max>, runtime_argument_bounds<_Tp>)
  -> dynamic_argument<_Arg, static_argument_bounds<_Min, _Max>>;
#endif // _CCCL_DOXYGEN_INVOKED

// =====================================================================
// deferred_argument
// =====================================================================

//! @brief Wraps a device-resident argument value that is not host-accessible at API call time.
template <class _Arg, class _StaticBounds = __no_bounds>
struct deferred_argument
{
  static_assert(::cuda::std::__is_cuda_std_span_v<_Arg>, "deferred_argument requires a cuda::std::span");

  using element_type = __argument_element_type_t<_Arg>;

  _Arg arg;
  _CCCL_NO_UNIQUE_ADDRESS _StaticBounds __static_bounds_;
  runtime_argument_bounds<element_type> __runtime_bounds_;

  _CCCL_API constexpr deferred_argument(_Arg __arg) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{}
  {}

  template <auto _Min, auto _Max>
  _CCCL_API constexpr deferred_argument(_Arg __arg, static_argument_bounds<_Min, _Max> __sb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{}
  {}

  template <class _BoundsTp>
  _CCCL_API constexpr deferred_argument(_Arg __arg, runtime_argument_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{static_cast<element_type>(__rb.min), static_cast<element_type>(__rb.max)}
  {}

  template <auto _Min, auto _Max, class _BoundsTp>
  _CCCL_API constexpr deferred_argument(
    _Arg __arg, static_argument_bounds<_Min, _Max> __sb, runtime_argument_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{static_cast<element_type>(__rb.min), static_cast<element_type>(__rb.max)}
  {}
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg>
_CCCL_HOST_DEVICE deferred_argument(_Arg) -> deferred_argument<_Arg>;

template <class _Arg, auto _Min, auto _Max>
_CCCL_HOST_DEVICE deferred_argument(_Arg, static_argument_bounds<_Min, _Max>)
  -> deferred_argument<_Arg, static_argument_bounds<_Min, _Max>>;

template <class _Arg, class _Tp>
_CCCL_HOST_DEVICE deferred_argument(_Arg, runtime_argument_bounds<_Tp>) -> deferred_argument<_Arg>;

template <class _Arg, auto _Min, auto _Max, class _Tp>
_CCCL_HOST_DEVICE deferred_argument(_Arg, static_argument_bounds<_Min, _Max>, runtime_argument_bounds<_Tp>)
  -> deferred_argument<_Arg, static_argument_bounds<_Min, _Max>>;
#endif // _CCCL_DOXYGEN_INVOKED

// =====================================================================
// __is_single_value_v
// =====================================================================

template <class _Tp>
inline constexpr bool __is_value_ref_v = false;
template <class _Tp>
inline constexpr bool __is_value_ref_v<::cuda::std::span<_Tp, 1>> = true;

template <class _Tp>
inline constexpr bool __is_single_value_v =
  ::cuda::std::is_arithmetic_v<::cuda::std::remove_cv_t<_Tp>> || ::cuda::std::is_enum_v<::cuda::std::remove_cv_t<_Tp>>
  || __is_value_ref_v<_Tp>;

// =====================================================================
// unwrap_argument
// =====================================================================

template <class _Tp>
inline constexpr bool __is_argument_wrapper_v = false;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_argument_wrapper_v<dynamic_argument<_Arg, _StaticBounds>> = true;
template <auto _Value>
inline constexpr bool __is_argument_wrapper_v<static_argument<_Value>> = true;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_argument_wrapper_v<deferred_argument<_Arg, _StaticBounds>> = true;

template <
  class _Tp,
  ::cuda::std::enable_if_t<!__is_argument_wrapper_v<::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<_Tp>>>,
                           int> = 0>
[[nodiscard]] _CCCL_API constexpr _Tp&& unwrap_argument(_Tp&& __arg) noexcept
{
  return ::cuda::std::forward<_Tp>(__arg);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg&
unwrap_argument(const dynamic_argument<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.arg;
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr const decltype(_Value)& unwrap_argument(const static_argument<_Value>&) noexcept
{
  return static_argument<_Value>::value;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg&
unwrap_argument(const deferred_argument<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.arg;
}

// =====================================================================
// Internal helpers for bounds computation
// =====================================================================

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __wrapper_static_min() noexcept
{
  if constexpr (::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
  {
    return ::cuda::std::numeric_limits<_ElementType>::lowest();
  }
  else
  {
    return static_cast<_ElementType>(_StaticBounds::min);
  }
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __wrapper_static_max() noexcept
{
  if constexpr (::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
  {
    return ::cuda::std::numeric_limits<_ElementType>::max();
  }
  else
  {
    return static_cast<_ElementType>(_StaticBounds::max);
  }
}

template <auto _Value>
inline constexpr auto __static_argument_compute_min = [] {
  using _VT = decltype(_Value);
  using _ET = __argument_element_type_t<_VT>;
  if constexpr (::cuda::std::is_same_v<_VT, _ET>)
  {
    return _Value;
  }
  else
  {
    auto __result = _Value[0];
    for (size_t __i = 1; __i < _Value.size(); ++__i)
    {
      if (_Value[__i] < __result)
      {
        __result = _Value[__i];
      }
    }
    return __result;
  }
}();

template <auto _Value>
inline constexpr auto __static_argument_compute_max = [] {
  using _VT = decltype(_Value);
  using _ET = __argument_element_type_t<_VT>;
  if constexpr (::cuda::std::is_same_v<_VT, _ET>)
  {
    return _Value;
  }
  else
  {
    auto __result = _Value[0];
    for (size_t __i = 1; __i < _Value.size(); ++__i)
    {
      if (_Value[__i] > __result)
      {
        __result = _Value[__i];
      }
    }
    return __result;
  }
}();

// =====================================================================
// argument_traits
// =====================================================================

//! @brief Traits for argument wrappers and plain argument values.
template <class _Tp>
struct argument_traits
{
  using value_type                         = _Tp;
  using element_type                       = __argument_element_type_t<_Tp>;
  static constexpr bool is_deferred        = false;
  static constexpr element_type static_min = ::cuda::std::numeric_limits<element_type>::lowest();
  static constexpr element_type static_max = ::cuda::std::numeric_limits<element_type>::max();
};

template <class _Arg, class _StaticBounds>
struct argument_traits<dynamic_argument<_Arg, _StaticBounds>>
{
  using value_type                         = _Arg;
  using element_type                       = __argument_element_type_t<_Arg>;
  static constexpr bool is_deferred        = false;
  static constexpr element_type static_min = __wrapper_static_min<element_type, _StaticBounds>();
  static constexpr element_type static_max = __wrapper_static_max<element_type, _StaticBounds>();
};

template <auto _Value>
struct argument_traits<static_argument<_Value>>
{
  using value_type                         = decltype(_Value);
  using element_type                       = __argument_element_type_t<value_type>;
  static constexpr bool is_deferred        = false;
  static constexpr element_type static_min = __static_argument_compute_min<_Value>;
  static constexpr element_type static_max = __static_argument_compute_max<_Value>;
};

template <class _Arg, class _StaticBounds>
struct argument_traits<deferred_argument<_Arg, _StaticBounds>>
{
  using value_type                         = _Arg;
  using element_type                       = __argument_element_type_t<_Arg>;
  static constexpr bool is_deferred        = true;
  static constexpr element_type static_min = __wrapper_static_min<element_type, _StaticBounds>();
  static constexpr element_type static_max = __wrapper_static_max<element_type, _StaticBounds>();
};

// =====================================================================
// argument_static_min / argument_static_max — free functions
// =====================================================================

//! @brief Returns the static minimum bound of an argument.
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr auto argument_static_min(_Tp) noexcept
{
  return argument_traits<_Tp>::static_min;
}

//! @brief Returns the static maximum bound of an argument.
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr auto argument_static_max(_Tp) noexcept
{
  return argument_traits<_Tp>::static_max;
}

// =====================================================================
// argument_min / argument_max
// =====================================================================

//! @brief Returns the runtime minimum bound of an argument.
template <class _Tp, ::cuda::std::enable_if_t<!__is_argument_wrapper_v<::cuda::std::remove_cv_t<_Tp>>, int> = 0>
[[nodiscard]] _CCCL_API constexpr auto argument_min(_Tp) noexcept
{
  return ::cuda::std::numeric_limits<__argument_element_type_t<_Tp>>::lowest();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto argument_min(static_argument<_Value>) noexcept
{
  return __static_argument_compute_min<_Value>;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto argument_min(dynamic_argument<_Arg, _StaticBounds> __arg) noexcept
{
  return __arg.__runtime_bounds_.min;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto argument_min(deferred_argument<_Arg, _StaticBounds> __arg) noexcept
{
  return __arg.__runtime_bounds_.min;
}

//! @brief Returns the runtime maximum bound of an argument.
template <class _Tp, ::cuda::std::enable_if_t<!__is_argument_wrapper_v<::cuda::std::remove_cv_t<_Tp>>, int> = 0>
[[nodiscard]] _CCCL_API constexpr auto argument_max(_Tp) noexcept
{
  return ::cuda::std::numeric_limits<__argument_element_type_t<_Tp>>::max();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto argument_max(static_argument<_Value>) noexcept
{
  return __static_argument_compute_max<_Value>;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto argument_max(dynamic_argument<_Arg, _StaticBounds> __arg) noexcept
{
  return __arg.__runtime_bounds_.max;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto argument_max(deferred_argument<_Arg, _StaticBounds> __arg) noexcept
{
  return __arg.__runtime_bounds_.max;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ARGUMENT_ARGUMENT_H
