//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_SHARED_MBARRIER_H
#define _CUDA___BARRIER_SHARED_MBARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  include <cuda/__ptx/instructions/mbarrier_arrive.h>
#  include <cuda/__ptx/instructions/mbarrier_check_layout.h>
#  include <cuda/__ptx/instructions/mbarrier_complete_tx.h>
#  include <cuda/__ptx/instructions/mbarrier_expect_tx.h>
#  include <cuda/__ptx/instructions/mbarrier_init.h>
#  include <cuda/__ptx/instructions/mbarrier_inval.h>
#  include <cuda/__ptx/instructions/mbarrier_wait.h>
#  include <cuda/__ptx/ptx_dot_variants.h>
#  include <cuda/__ptx/ptx_helper_functions.h>
#endif // _CCCL_CUDA_COMPILATION()
#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/high_resolution_clock.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/cstdint>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
struct __mbarrier_wait_status
{
  bool __complete;
  bool __report_predicate;
  ::cuda::std::uint8_t __report_value;
};

class __shared_barrier_storage
{
  ::cuda::std::uint64_t __barrier_;

public:
  using __arrival_token = ::cuda::std::uint64_t;

  _CCCL_HIDE_FROM_ABI __shared_barrier_storage() = default;

  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint64_t& __storage_ref() noexcept
  {
    return __barrier_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API const ::cuda::std::uint64_t& __storage_ref() const noexcept
  {
    return __barrier_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint64_t* __native_handle() const noexcept
  {
    return const_cast<::cuda::std::uint64_t*>(&__barrier_);
  }

#if _CCCL_CUDA_COMPILATION()
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __init(::cuda::std::uint32_t __count) const
  {
    ::cuda::ptx::mbarrier_init(__native_handle(), __count);
  }

#  if __cccl_ptx_isa >= 940
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __init_status_reporting(::cuda::std::uint32_t __count) const
  {
    ::cuda::ptx::mbarrier_init(::cuda::ptx::layout_v1, __native_handle(), __count);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __has_status_reporting_layout() const
  {
    return ::cuda::ptx::mbarrier_check_layout(::cuda::ptx::layout_v1, __native_handle());
  }
#  endif // __cccl_ptx_isa >= 940

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __inval() const
  {
    ::cuda::ptx::mbarrier_inval(__native_handle());
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE __arrival_token __arrive(::cuda::std::ptrdiff_t __update) const
  {
    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (return ::cuda::ptx::mbarrier_arrive(__native_handle(), static_cast<::cuda::std::uint32_t>(__update));))

    NV_IF_TARGET(
      NV_PROVIDES_SM_80,
      (
        // Need 2 instructions, can't finish barrier with arrive > 1.
        if (__update > 1) {
          ::cuda::ptx::mbarrier_arrive_no_complete(__native_handle(), static_cast<::cuda::std::uint32_t>(__update - 1));
        } return ::cuda::ptx::mbarrier_arrive(__native_handle());))

    _CCCL_UNREACHABLE();
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __test_wait(__arrival_token __token) const
  {
    return ::cuda::ptx::mbarrier_test_wait(__native_handle(), __token);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __test_wait_phase(::cuda::std::uint32_t __phase) const
  {
    return ::cuda::ptx::mbarrier_test_wait_parity(__native_handle(), __phase);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __test_wait_parity(bool __phase_parity) const
  {
    return __test_wait_phase(__phase_parity ? 1u : 0u);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __try_wait(__arrival_token __token) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return ::cuda::ptx::mbarrier_try_wait(__native_handle(), __token);))

    NV_IF_TARGET(NV_PROVIDES_SM_80, (return __test_wait(__token);))

    _CCCL_UNREACHABLE();
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __try_wait_sm90(__arrival_token __token, ::cuda::std::uint32_t __suspend_time_hint) const
  {
    return ::cuda::ptx::mbarrier_try_wait(__native_handle(), __token, __suspend_time_hint);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __try_wait_for(__arrival_token __token, ::cuda::std::chrono::nanoseconds __nanosec) const
  {
    if (__nanosec.count() < 1)
    {
      return __try_wait(__token);
    }

    bool __ready = false;
    const ::cuda::std::chrono::high_resolution_clock::time_point __start =
      ::cuda::std::chrono::high_resolution_clock::now();
    ::cuda::std::chrono::nanoseconds __elapsed(0);
    do
    {
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_90,
        (const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
         __ready                                 = __try_wait_sm90(__token, __wait_nsec);),
        (__ready = __test_wait(__token);))
      __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
    } while (!__ready && (__nanosec > __elapsed));
    return __ready;
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __try_wait_phase(::cuda::std::uint32_t __phase) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return ::cuda::ptx::mbarrier_try_wait_parity(__native_handle(), __phase);))

    NV_IF_TARGET(NV_PROVIDES_SM_80, (return __test_wait_phase(__phase);))

    _CCCL_UNREACHABLE();
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __try_wait_phase_sm90(::cuda::std::uint32_t __phase, ::cuda::std::uint32_t __suspend_time_hint) const
  {
    return ::cuda::ptx::mbarrier_try_wait_parity(__native_handle(), __phase, __suspend_time_hint);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __try_wait_phase_for(::cuda::std::uint32_t __phase, ::cuda::std::chrono::nanoseconds __nanosec) const
  {
    if (__nanosec.count() < 1)
    {
      return __try_wait_phase(__phase);
    }

    bool __ready = false;
    const ::cuda::std::chrono::high_resolution_clock::time_point __start =
      ::cuda::std::chrono::high_resolution_clock::now();
    ::cuda::std::chrono::nanoseconds __elapsed(0);
    do
    {
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_90,
        (const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
         __ready                                 = __try_wait_phase_sm90(__phase, __wait_nsec);),
        (__ready = __test_wait_phase(__phase);))
      __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
    } while (!__ready && (__nanosec > __elapsed));
    return __ready;
  }

#  if __cccl_ptx_isa >= 940
  [[nodiscard]] _CCCL_DEVICE_API
  _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status __test_wait_status(__arrival_token __token) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_test_wait(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __token);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status
  __try_wait_status(__arrival_token __token) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_try_wait(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __token);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status
  __try_wait_status(__arrival_token __token, ::cuda::std::uint32_t __suspend_time_hint) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_try_wait(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __token,
      __suspend_time_hint);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status
  __test_wait_phase_status(::cuda::std::uint32_t __phase) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_test_wait_parity(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __phase);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status
  __try_wait_phase_status(::cuda::std::uint32_t __phase) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_try_wait_parity(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __phase);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::__detail::__mbarrier_wait_status
  __try_wait_phase_status(::cuda::std::uint32_t __phase, ::cuda::std::uint32_t __suspend_time_hint) const
  {
    bool __report_predicate             = false;
    ::cuda::std::uint8_t __report_value = 0;
    const bool __complete               = ::cuda::ptx::mbarrier_try_wait_parity(
      ::cuda::ptx::mbarrier_phase_primary,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __report_predicate,
      __report_value,
      __native_handle(),
      __phase,
      __suspend_time_hint);
    return {__complete, __report_predicate, __report_value};
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __test_wait_conditional_phase(::cuda::std::uint32_t __phase) const
  {
    return ::cuda::ptx::mbarrier_test_wait_parity(
      ::cuda::ptx::mbarrier_phase_conditional,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __native_handle(),
      __phase);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __try_wait_conditional_phase(::cuda::std::uint32_t __phase) const
  {
    return ::cuda::ptx::mbarrier_try_wait_parity(
      ::cuda::ptx::mbarrier_phase_conditional,
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      __native_handle(),
      __phase);
  }
#  endif // __cccl_ptx_isa >= 940

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __try_wait_parity(bool __phase_parity) const
  {
    return __try_wait_phase(__phase_parity ? 1u : 0u);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  __try_wait_parity_for(bool __phase_parity, ::cuda::std::chrono::nanoseconds __nanosec) const
  {
    return __try_wait_phase_for(__phase_parity ? 1u : 0u, __nanosec);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __arrive_and_drop() const
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      ((void) ::cuda::ptx::mbarrier_arrive_drop(
         ::cuda::ptx::sem_release, ::cuda::ptx::scope_cta, ::cuda::ptx::space_shared, __native_handle(), 1);),
      // The generated SM80 wrapper uses the noComplete modifier, which is not equivalent to arrive_and_drop.
      (asm volatile("mbarrier.arrive_drop.shared.b64 _, [%0];" ::"r"(
        static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(__native_handle()))) : "memory");))
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __expect_tx(::cuda::std::ptrdiff_t __transaction_count_update) const
  {
    ::cuda::ptx::mbarrier_expect_tx(
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      ::cuda::ptx::space_shared,
      __native_handle(),
      static_cast<::cuda::std::uint32_t>(__transaction_count_update));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE __arrival_token
  __arrive_tx(::cuda::std::ptrdiff_t __arrive_count_update, ::cuda::std::ptrdiff_t __transaction_count_update) const
  {
    if (__arrive_count_update == 1)
    {
      return ::cuda::ptx::mbarrier_arrive_expect_tx(
        ::cuda::ptx::sem_release,
        ::cuda::ptx::scope_cta,
        ::cuda::ptx::space_shared,
        __native_handle(),
        static_cast<::cuda::std::uint32_t>(__transaction_count_update));
    }

    __expect_tx(__transaction_count_update);
    return __arrive(__arrive_count_update);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __complete_tx(::cuda::std::ptrdiff_t __transaction_count_update) const
  {
    ::cuda::ptx::mbarrier_complete_tx(
      ::cuda::ptx::sem_relaxed,
      ::cuda::ptx::scope_cta,
      ::cuda::ptx::space_shared,
      __native_handle(),
      static_cast<::cuda::std::uint32_t>(__transaction_count_update));
  }
#endif // _CCCL_CUDA_COMPILATION()
};

static_assert(sizeof(__shared_barrier_storage) == sizeof(::cuda::std::uint64_t),
              "shared barrier storage must remain a single mbarrier word");
static_assert(alignof(__shared_barrier_storage) == alignof(::cuda::std::uint64_t),
              "shared barrier storage must keep uint64_t alignment");
} // namespace __detail

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BARRIER_SHARED_MBARRIER_H
