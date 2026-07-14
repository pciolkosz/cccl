//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___KERNEL_JIT_DISPATCH
#define _CUDAX___KERNEL_JIT_DISPATCH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::prototype
{
using jit_kernel_id = ::cuda::std::uint32_t;

template <class... _Ts>
using jit_packed_tuple = ::cuda::std::tuple<_Ts...>;

template <class _Kernel>
struct jit_kernel_source;

namespace __detail
{
template <class _Impl, class = void>
struct __has_explicit_packed_tuple
{
  static constexpr bool __value = false;
};

template <class _Impl>
struct __has_explicit_packed_tuple<_Impl, ::cuda::std::void_t<typename _Impl::packed_tuple_type>>
{
  static constexpr bool __value = true;
};

template <class _CallOp>
struct __packed_tuple_from_call_op;

template <class _Class, class _Ret, class... _Args>
struct __packed_tuple_from_call_op<_Ret (_Class::*)(_Args...)>
{
  using __type = jit_packed_tuple<_Args...>;
};

template <class _Class, class _Ret, class... _Args>
struct __packed_tuple_from_call_op<_Ret (_Class::*)(_Args...) const>
{
  using __type = jit_packed_tuple<_Args...>;
};

template <class _Class, class _Ret, class... _Args>
struct __packed_tuple_from_call_op<_Ret (_Class::*)(_Args...) volatile>
{
  using __type = jit_packed_tuple<_Args...>;
};

template <class _Class, class _Ret, class... _Args>
struct __packed_tuple_from_call_op<_Ret (_Class::*)(_Args...) const volatile>
{
  using __type = jit_packed_tuple<_Args...>;
};

template <class _Impl, bool = __has_explicit_packed_tuple<_Impl>::__value>
struct __packed_tuple_for;

template <class _Impl>
struct __packed_tuple_for<_Impl, true>
{
  using __type = typename _Impl::packed_tuple_type;
};

template <class _Impl>
struct __packed_tuple_for<_Impl, false>
{
  using __type = typename __packed_tuple_from_call_op<decltype(&_Impl::operator())>::__type;
};

template <class _Impl>
using __packed_tuple_t = typename __packed_tuple_for<_Impl>::__type;

template <::cuda::std::size_t _Value, ::cuda::std::size_t... _Rest>
struct __static_max;

template <::cuda::std::size_t _Value>
struct __static_max<_Value>
{
  static constexpr ::cuda::std::size_t __value = _Value;
};

template <::cuda::std::size_t _Value, ::cuda::std::size_t _Next, ::cuda::std::size_t... _Rest>
struct __static_max<_Value, _Next, _Rest...>
{
  static constexpr ::cuda::std::size_t __tail  = __static_max<_Next, _Rest...>::__value;
  static constexpr ::cuda::std::size_t __value = _Value < __tail ? __tail : _Value;
};

template <class... _Impls>
inline constexpr ::cuda::std::size_t __max_packed_size_v =
  __static_max<sizeof(__packed_tuple_t<_Impls>)...>::__value;

template <class... _Impls>
inline constexpr ::cuda::std::size_t __max_packed_align_v =
  __static_max<alignof(__packed_tuple_t<_Impls>)...>::__value;

template <jit_kernel_id _Id, class... _Impls>
struct __impl_for_id;

template <jit_kernel_id _Id>
struct __impl_for_id<_Id>
{
  static_assert(_Id != _Id, "No jit_kernel implementation has the requested id.");
};

template <bool _Matches, jit_kernel_id _Id, class _Impl, class... _Rest>
struct __impl_for_id_select;

template <jit_kernel_id _Id, class _Impl, class... _Rest>
struct __impl_for_id_select<true, _Id, _Impl, _Rest...>
{
  using __type = _Impl;
};

template <jit_kernel_id _Id, class _Impl, class... _Rest>
struct __impl_for_id_select<false, _Id, _Impl, _Rest...>
{
  using __type = typename __impl_for_id<_Id, _Rest...>::__type;
};

template <jit_kernel_id _Id, class _Impl, class... _Rest>
struct __impl_for_id<_Id, _Impl, _Rest...>
{
  using __type = typename __impl_for_id_select<_Id == _Impl::id, _Id, _Impl, _Rest...>::__type;
};

template <jit_kernel_id _Id, class... _Impls>
struct __count_id;

template <jit_kernel_id _Id>
struct __count_id<_Id>
{
  static constexpr int __value = 0;
};

template <jit_kernel_id _Id, class _Impl, class... _Rest>
struct __count_id<_Id, _Impl, _Rest...>
{
  static constexpr int __value = (_Id == _Impl::id ? 1 : 0) + __count_id<_Id, _Rest...>::__value;
};

template <class... _Impls>
struct __ids_are_unique;

template <>
struct __ids_are_unique<>
{
  static constexpr bool __value = true;
};

template <class _Impl, class... _Rest>
struct __ids_are_unique<_Impl, _Rest...>
{
  static constexpr bool __value =
    (__count_id<_Impl::id, _Impl, _Rest...>::__value == 1) && __ids_are_unique<_Rest...>::__value;
};

template <::cuda::std::size_t _Size, ::cuda::std::size_t _Align>
struct alignas(_Align) __erased_arg_storage
{
  unsigned char __bytes_[_Size == 0 ? 1 : _Size]{};

#if !_CCCL_COMPILER(NVRTC)
  template <class _PackedTuple>
  _CCCL_HOST_API void __store(const _PackedTuple& __packed) noexcept
  {
    static_assert(sizeof(_PackedTuple) <= _Size);
    static_assert(alignof(_PackedTuple) <= _Align);
    static_assert(::cuda::std::is_trivially_copyable_v<_PackedTuple>);

    const auto* __src = reinterpret_cast<const unsigned char*>(&__packed);
    for (::cuda::std::size_t __i = 0; __i != sizeof(_PackedTuple); ++__i)
    {
      __bytes_[__i] = __src[__i];
    }
  }
#endif // !_CCCL_COMPILER(NVRTC)

  template <class _PackedTuple>
  [[nodiscard]] _CCCL_HOST_DEVICE_API const _PackedTuple& __as() const noexcept
  {
    static_assert(sizeof(_PackedTuple) <= _Size);
    static_assert(alignof(_PackedTuple) <= _Align);
    static_assert(::cuda::std::is_trivially_copyable_v<_PackedTuple>);
    return *reinterpret_cast<const _PackedTuple*>(__bytes_);
  }
};

template <class... _Impls>
struct __dispatcher_invoker;

template <>
struct __dispatcher_invoker<>
{
  template <class _Dispatcher>
  _CCCL_HOST_DEVICE_API static bool __dispatch(const _Dispatcher&) noexcept
  {
    return false;
  }
};

template <class _Impl, class... _Rest>
struct __dispatcher_invoker<_Impl, _Rest...>
{
  template <class _Dispatcher>
  _CCCL_HOST_DEVICE_API static bool __dispatch(const _Dispatcher& __dispatcher)
  {
    if (__dispatcher.id() == _Impl::id)
    {
      ::cuda::std::apply(_Impl{}, __dispatcher.template packed_as<_Impl>());
      return true;
    }
    return __dispatcher_invoker<_Rest...>::__dispatch(__dispatcher);
  }
};

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp, class... _Args>
struct __is_brace_constructible
{
  template <class _Up, class... _Us, class = decltype(_Up{::cuda::std::declval<_Us>()...})>
  static char __test(int);

  template <class...>
  static long __test(...);

  static constexpr bool __value = sizeof(__test<_Tp, _Args...>(0)) == sizeof(char);
};

template <class... _Impls>
struct __runtime_dispatcher_packer;

template <>
struct __runtime_dispatcher_packer<>
{
  template <class _Dispatcher, class... _Args>
  _CCCL_HOST_API static bool __pack(jit_kernel_id, _Dispatcher&, _Args&&...) noexcept
  {
    return false;
  }
};

template <class _Impl, class... _Rest>
struct __runtime_dispatcher_packer<_Impl, _Rest...>
{
  template <class _Dispatcher, class... _Args>
  _CCCL_HOST_API static bool __pack(jit_kernel_id __id, _Dispatcher& __dispatcher, _Args&&... __args)
  {
    if (__id == _Impl::id)
    {
      using _PackedTuple = __packed_tuple_t<_Impl>;
      static_assert(__is_brace_constructible<_PackedTuple, _Args...>::__value,
                    "Runtime-id jit_dispatcher packing requires every candidate implementation to accept these "
                    "arguments.");

      __dispatcher.template __set<_Impl>(::cuda::std::forward<_Args>(__args)...);
      return true;
    }
    return __runtime_dispatcher_packer<_Rest...>::__pack(__id, __dispatcher, ::cuda::std::forward<_Args>(__args)...);
  }
};
#endif // !_CCCL_COMPILER(NVRTC)
} // namespace __detail

template <class... _Impls>
class jit_dispatcher
{
  static_assert(sizeof...(_Impls) != 0, "jit_dispatcher needs at least one implementation functor.");
  static_assert(__detail::__ids_are_unique<_Impls...>::__value, "jit_dispatcher implementation ids must be unique.");
  static_assert((::cuda::std::is_trivially_copyable_v<__detail::__packed_tuple_t<_Impls>> && ...),
                "jit_dispatcher implementation packed tuple types must be trivially copyable.");

  using _Storage =
    __detail::__erased_arg_storage<__detail::__max_packed_size_v<_Impls...>,
                                   __detail::__max_packed_align_v<_Impls...>>;

public:
  jit_dispatcher() = default;

  inline static constexpr ::cuda::std::size_t max_packed_size  = __detail::__max_packed_size_v<_Impls...>;
  inline static constexpr ::cuda::std::size_t max_packed_align = __detail::__max_packed_align_v<_Impls...>;

  template <class _Impl>
  inline static constexpr bool contains_implementation_v = (::cuda::std::is_same_v<_Impl, _Impls> || ...);

#if !_CCCL_COMPILER(NVRTC)
  template <jit_kernel_id _Id, class... _Args>
  [[nodiscard]] _CCCL_HOST_API static jit_dispatcher make_dispatcher(_Args&&... __args)
  {
    using _Impl        = typename __detail::__impl_for_id<_Id, _Impls...>::__type;
    using _PackedTuple = __detail::__packed_tuple_t<_Impl>;

    static_assert(__detail::__is_brace_constructible<_PackedTuple, _Args...>::__value,
                  "The selected jit_dispatcher implementation cannot be packed from these arguments.");

    jit_dispatcher __dispatcher;
    __dispatcher.template __set<_Impl>(::cuda::std::forward<_Args>(__args)...);
    return __dispatcher;
  }

  template <class... _Args>
  [[nodiscard]] _CCCL_HOST_API static jit_dispatcher make_dispatcher(jit_kernel_id __id, _Args&&... __args)
  {
    jit_dispatcher __dispatcher;
    const bool __matched =
      __detail::__runtime_dispatcher_packer<_Impls...>::__pack(
        __id, __dispatcher, ::cuda::std::forward<_Args>(__args)...);
    _CCCL_ASSERT(__matched, "Runtime jit_dispatcher id does not match any implementation.");
    return __dispatcher;
  }
#endif // !_CCCL_COMPILER(NVRTC)

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr jit_kernel_id id() const noexcept
  {
    return __id_;
  }

  template <class _Impl>
  [[nodiscard]] _CCCL_HOST_DEVICE_API const __detail::__packed_tuple_t<_Impl>& packed_as() const noexcept
  {
    static_assert(contains_implementation_v<_Impl>, "Implementation does not belong to this jit_dispatcher.");
    return __storage_.template __as<__detail::__packed_tuple_t<_Impl>>();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static bool dispatch(const jit_dispatcher& __dispatcher)
  {
    return __detail::__dispatcher_invoker<_Impls...>::__dispatch(__dispatcher);
  }

  _CCCL_DEVICE_API void operator()() const
  {
    (void) dispatch(*this);
  }

private:
#if !_CCCL_COMPILER(NVRTC)
  template <class...>
  friend struct __detail::__runtime_dispatcher_packer;

  template <class _Impl, class... _Args>
  _CCCL_HOST_API void __set(_Args&&... __args)
  {
    static_assert(contains_implementation_v<_Impl>, "Implementation does not belong to this jit_dispatcher.");

    using _PackedTuple = __detail::__packed_tuple_t<_Impl>;
    _PackedTuple __packed{::cuda::std::forward<_Args>(__args)...};

    __id_ = _Impl::id;
    __storage_.template __store<_PackedTuple>(__packed);
  }
#endif // !_CCCL_COMPILER(NVRTC)

  jit_kernel_id __id_{};
  _Storage __storage_{};
};

template <class... _Impls>
class jit_kernel
{
public:
  using dispatcher_type = jit_dispatcher<_Impls...>;

  inline static constexpr ::cuda::std::size_t max_packed_size  = dispatcher_type::max_packed_size;
  inline static constexpr ::cuda::std::size_t max_packed_align = dispatcher_type::max_packed_align;

  template <class _Impl>
  inline static constexpr bool contains_implementation_v = dispatcher_type::template contains_implementation_v<_Impl>;

#if !_CCCL_COMPILER(NVRTC)
  template <jit_kernel_id _Id, class... _Args>
  [[nodiscard]] _CCCL_HOST_API static dispatcher_type make_dispatcher(_Args&&... __args)
  {
    return dispatcher_type::template make_dispatcher<_Id>(::cuda::std::forward<_Args>(__args)...);
  }

  template <class... _Args>
  [[nodiscard]] _CCCL_HOST_API static dispatcher_type make_dispatcher(jit_kernel_id __id, _Args&&... __args)
  {
    return dispatcher_type::make_dispatcher(__id, ::cuda::std::forward<_Args>(__args)...);
  }
#endif // !_CCCL_COMPILER(NVRTC)

  [[nodiscard]] _CCCL_HOST_DEVICE_API static bool dispatch(const dispatcher_type& __dispatcher)
  {
    return dispatcher_type::dispatch(__dispatcher);
  }
};
} // namespace cuda::experimental::prototype

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___KERNEL_JIT_DISPATCH
