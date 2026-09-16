//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_SHARED_BARRIER_H
#define _CUDA___BARRIER_SHARED_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/shared_mbarrier.h>
#include <cuda/__fwd/barrier.h>
#include <cuda/__utility/status_policy.h>
#if _CCCL_CUDA_COMPILATION()
#  include <cuda/__memory/address_space.h>
#endif // _CCCL_CUDA_COMPILATION()
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/high_resolution_clock.h>
#include <cuda/std/__chrono/time_point.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/cstdint>

#include <nv/target>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 3)
#  include <cuda_runtime_api.h>
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 3)

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 3)

_CCCL_BEGIN_NAMESPACE_CUDA

enum class shared_barrier_kind
{
  completion_only,
  status_reporting,
};

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint64_t* barrier_native_handle(::cuda::shared_barrier& __b);
_CCCL_END_NAMESPACE_CUDA_DEVICE

_CCCL_BEGIN_NAMESPACE_CUDA

class shared_barrier : private ::cuda::__detail::__shared_barrier_storage
{
  _CCCL_DEVICE_API friend ::cuda::std::uint64_t* ::cuda::device::_LIBCUDACXX_ABI_NAMESPACE::barrier_native_handle(
    ::cuda::shared_barrier& __b);

public:
  class operation_status
  {
    bool __complete_                     = false;
    bool __report_predicate_             = false;
    ::cuda::std::uint8_t __report_value_ = 0;
    mutable bool __report_inspected_     = false;

    _CCCL_HOST_DEVICE_API constexpr operation_status(
      bool __complete, bool __report_predicate, ::cuda::std::uint8_t __report_value) noexcept
        : __complete_(__complete)
        , __report_predicate_(__report_predicate)
        , __report_value_(__report_value)
    {}

    friend class shared_barrier;

    _CCCL_HOST_DEVICE_API void __assert_report_inspected() const noexcept
    {
      if (__report_predicate_ && !__report_inspected_)
      {
        NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
        _CCCL_UNREACHABLE();
      }
    }

    _CCCL_HOST_DEVICE_API void __mark_report_inspected() const noexcept
    {
      __report_inspected_ = true;
    }

  public:
    _CCCL_HOST_DEVICE_API constexpr operation_status() noexcept {}

    operation_status(const operation_status&)            = delete;
    operation_status& operator=(const operation_status&) = delete;

    _CCCL_HOST_DEVICE_API operation_status(operation_status&& __other) noexcept
        : __complete_(__other.__complete_)
        , __report_predicate_(__other.__report_predicate_)
        , __report_value_(__other.__report_value_)
        , __report_inspected_(__other.__report_inspected_)
    {
      __other.__report_predicate_ = false;
      __other.__report_inspected_ = true;
    }

    _CCCL_HOST_DEVICE_API operation_status& operator=(operation_status&& __other) noexcept
    {
      __assert_report_inspected();
      __complete_                 = __other.__complete_;
      __report_predicate_         = __other.__report_predicate_;
      __report_value_             = __other.__report_value_;
      __report_inspected_         = __other.__report_inspected_;
      __other.__report_predicate_ = false;
      __other.__report_inspected_ = true;
      return *this;
    }

    _CCCL_HOST_DEVICE_API ~operation_status() noexcept
    {
      __assert_report_inspected();
    }

    [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool complete() const noexcept
    {
      return __complete_;
    }

    [[nodiscard]] _CCCL_HOST_DEVICE_API bool has_report() const noexcept
    {
      if (__report_predicate_)
      {
        __mark_report_inspected();
      }
      return __report_predicate_;
    }

  private:
    _CCCL_DEVICE_API static void __assert_fabric_status(::cudaError_t __status) noexcept
    {
      _CCCL_ASSERT(__status == ::cudaSuccess, "failed to decode shared_barrier status");
      if (__status != ::cudaSuccess)
      {
        ::cuda::std::terminate();
      }
    }

    [[nodiscard]] _CCCL_HOST_DEVICE_API static bool __encodes_fabric_errors(status_source __source) noexcept
    {
      switch (__source)
      {
        case status_source::generic_fabric:
          return true;
      }
      _CCCL_UNREACHABLE();
    }

    [[nodiscard]] _CCCL_DEVICE_API static ::cudaFabricOpStatusSource
    __cuda_status_source(status_source __source) noexcept
    {
      switch (__source)
      {
        case status_source::generic_fabric:
          return ::cudaFabricOpStatusSourceMbarrierV1;
      }
      _CCCL_UNREACHABLE();
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned int __error_count(status_source __source) const noexcept
    {
      if (!__encodes_fabric_errors(__source))
      {
        return 0;
      }
      unsigned int __count = 0;
      auto __report_value  = __report_value_;
      __assert_fabric_status(::cudaFabricOpErrorStatusCount(&__report_value, __cuda_status_source(__source), &__count));
      return __count;
    }

    [[nodiscard]] _CCCL_DEVICE_API ::cudaFabricOpStatusInfo
    __error_status(status_source __source, unsigned int __status_index) const noexcept
    {
      _CCCL_ASSERT(__encodes_fabric_errors(__source), "shared_barrier status source does not encode fabric errors");
      ::cudaFabricOpStatusInfo __status_info{};
      auto __report_value = __report_value_;
      __assert_fabric_status(
        ::cudaFabricOpErrorStatusGet(&__report_value, __cuda_status_source(__source), __status_index, &__status_info));
      return __status_info;
    }

  public:
    [[nodiscard]] _CCCL_DEVICE_API unsigned int get_error_count(status_source __source) const noexcept
    {
      __mark_report_inspected();
      return __error_count(__source);
    }

    template <class _Fn>
    _CCCL_DEVICE_API void for_each_error(status_source __source, _Fn __fn) const noexcept
    {
      __mark_report_inspected();
      const unsigned int __count = __error_count(__source);
      for (unsigned int __index = 0; __index != __count; ++__index)
      {
        __fn(__error_status(__source, __index));
      }
    }

    [[nodiscard]] _CCCL_DEVICE_API status_action classify(status_source __source) const noexcept
    {
      __mark_report_inspected();
      _CCCL_ASSERT(__report_predicate_, "cannot classify a shared_barrier operation_status without a report");
      if (!__report_predicate_)
      {
        return status_action::abort;
      }
      return __error_count(__source) == 0 ? status_action::retry : status_action::abort;
    }
  };

  class arrival_token
  {
    ::cuda::std::uint64_t __token_ = 0;

    _CCCL_HOST_DEVICE_API explicit constexpr arrival_token(::cuda::std::uint64_t __token) noexcept
        : __token_(__token)
    {}

    friend class shared_barrier;

  public:
    arrival_token() = delete;

    _CCCL_HOST_DEVICE_API constexpr arrival_token(const arrival_token& __other) noexcept
        : __token_(__other.__token_)
    {}

    _CCCL_HOST_DEVICE_API constexpr arrival_token(arrival_token&& __other) noexcept
        : __token_(__other.__token_)
    {}

    _CCCL_HOST_DEVICE_API constexpr arrival_token& operator=(const arrival_token& __other) noexcept
    {
      __token_ = __other.__token_;
      return *this;
    }

    _CCCL_HOST_DEVICE_API constexpr arrival_token& operator=(arrival_token&& __other) noexcept
    {
      __token_ = __other.__token_;
      return *this;
    }
  };

  _CCCL_HIDE_FROM_ABI shared_barrier() = default;

  shared_barrier(const shared_barrier&)            = delete;
  shared_barrier& operator=(const shared_barrier&) = delete;

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint64_t* native_handle() noexcept
  {
    return __native_handle();
  }

private:
  [[noreturn]] _CCCL_HOST_DEVICE_API static void __unsupported_storage() noexcept
  {
    _CCCL_ASSERT(false, "shared_barrier requires local shared memory and shared-memory mbarrier support");
    NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
    _CCCL_UNREACHABLE();
  }

  [[noreturn]] _CCCL_HOST_DEVICE_API static void __unsupported_status_reporting() noexcept
  {
    _CCCL_ASSERT(false, "shared_barrier status_reporting kind requires mbarrier layout v1 support");
    NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
    _CCCL_UNREACHABLE();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::ptrdiff_t
  __max_for_kind(shared_barrier_kind __kind) noexcept
  {
    return __kind == shared_barrier_kind::status_reporting ? ((1 << 9) - 1) : ((1 << 20) - 1);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::ptrdiff_t __max_transaction_count_update() noexcept
  {
    return (1 << 20) - 1;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::uint64_t
  __token_value(arrival_token __token) noexcept
  {
    return __token.__token_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static ::cuda::std::uint32_t __phase_value(int __phase) noexcept
  {
    _CCCL_ASSERT(0 <= __phase, "Phase number must be non-negative.");
    return static_cast<::cuda::std::uint32_t>(__phase);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __assert_supported_storage() const
  {
    if (!::cuda::device::is_object_from(__storage_ref(), ::cuda::device::address_space::shared))
    {
      NV_IF_TARGET(
        NV_PROVIDES_SM_90,
        (_CCCL_ASSERT(!::cuda::device::is_object_from(__storage_ref(), ::cuda::device::address_space::cluster_shared),
                      "shared_barrier must not be in another block's cluster shared memory");))
      __unsupported_storage();
    }
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static operation_status
  __make_operation_status(::cuda::__detail::__mbarrier_wait_status __result) noexcept
  {
    return operation_status(__result.__complete, __result.__report_predicate, __result.__report_value);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE operation_status
  __make_completion_only_operation_status(bool __complete) const noexcept
  {
    return operation_status(__complete, false, 0);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE operation_status
  __test_wait_completion_only(arrival_token __token) const
  {
    return __make_completion_only_operation_status(__test_wait(__token_value(__token)));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE operation_status __test_wait_phase_completion_only(int __phase) const
  {
    return __make_completion_only_operation_status(__test_wait_phase(__phase_value(__phase)));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE operation_status
  __try_wait_completion_only(arrival_token __token) const
  {
    return __make_completion_only_operation_status(__try_wait(__token_value(__token)));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE operation_status __try_wait_phase_completion_only(int __phase) const
  {
    return __make_completion_only_operation_status(__try_wait_phase(__phase_value(__phase)));
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::ptrdiff_t __max_for_current_kind() const
  {
#  if __cccl_ptx_isa >= 940
    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (return __max_for_kind(is_kind(shared_barrier_kind::status_reporting) ? shared_barrier_kind::status_reporting
                                                                            : shared_barrier_kind::completion_only);))
#  endif // __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return __max_for_kind(shared_barrier_kind::completion_only);))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool __completed_ignoring_report(const operation_status& __status) const
  {
    (void) __status.has_report();
    return __status.complete();
  }

public:
  _CCCL_HOST_DEVICE_API ~shared_barrier()
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80, (__inval(); return;))

    __unsupported_storage();
  }

  _CCCL_HOST_DEVICE_API inline friend void
  init(shared_barrier* __b, shared_barrier_kind __kind, ::cuda::std::ptrdiff_t __expected)
  {
    _CCCL_ASSERT(1 <= __expected, "Expected arrival count must be at least one.");
    _CCCL_ASSERT(__expected <= shared_barrier::max(__kind),
                 "Expected arrival count cannot exceed shared_barrier::max(kind).");

#  if __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (
                   __b->__assert_supported_storage(); if (__kind == shared_barrier_kind::status_reporting) {
                     __b->__init_status_reporting(static_cast<::cuda::std::uint32_t>(__expected));
                   } else { __b->__init(static_cast<::cuda::std::uint32_t>(__expected)); } return;))
#  endif // __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (__b->__assert_supported_storage(); if (__kind == shared_barrier_kind::status_reporting) {
                   __unsupported_status_reporting();
                 } __b->__init(static_cast<::cuda::std::uint32_t>(__expected));
                  return;))

    __unsupported_storage();
  }

  _CCCL_HOST_DEVICE_API inline friend void init(shared_barrier* __b, ::cuda::std::ptrdiff_t __expected)
  {
    init(__b, shared_barrier_kind::completion_only, __expected);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool is_kind(shared_barrier_kind __kind) const
  {
#  if __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (const bool __is_v1 = __has_status_reporting_layout();
                  return __kind == shared_barrier_kind::status_reporting ? __is_v1 : !__is_v1;))
#  endif // __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return __kind == shared_barrier_kind::completion_only;))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API arrival_token arrive(::cuda::std::ptrdiff_t __update = 1)
  {
    _CCCL_ASSERT(1 <= __update, "Arrival count update must be at least one.");
    _CCCL_ASSERT(__update <= __max_for_current_kind(),
                 "Arrival count update cannot exceed shared_barrier::max(active kind).");

    NV_IF_TARGET(NV_PROVIDES_SM_80, (return arrival_token(__arrive(__update));))

    __unsupported_storage();
  }

  _CCCL_HOST_DEVICE_API void expect_tx(::cuda::std::ptrdiff_t __transaction_count_update);
  [[nodiscard]] _CCCL_HOST_DEVICE_API arrival_token
  arrive_tx(::cuda::std::ptrdiff_t __arrive_count_update, ::cuda::std::ptrdiff_t __transaction_count_update);

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status test_wait(arrival_token __token, return_status_t) const
  {
#  if __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __make_operation_status(__test_wait_status(__token_value(__token)));))
#  endif // __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return __test_wait_completion_only(__token);))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool test_wait(arrival_token __token, ignore_status_t) const
  {
    const operation_status __status = test_wait(__token, return_status);
    return __completed_ignoring_report(__status);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait(arrival_token __token, return_status_t) const
  {
#  if __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __make_operation_status(__try_wait_status(__token_value(__token)));))
#  endif // __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return __try_wait_completion_only(__token);))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait(arrival_token __token, ignore_status_t) const
  {
    const operation_status __status = try_wait(__token, return_status);
    return __completed_ignoring_report(__status);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status wait(arrival_token __token, return_status_t) const
  {
    operation_status __result;
    do
    {
      __result = try_wait(__token, return_status);
    } while (!__result.complete());
    return __result;
  }

  _CCCL_HOST_DEVICE_API void wait(arrival_token __token, ignore_status_t) const
  {
    while (!try_wait(__token, ignore_status))
    {
    }
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status arrive_and_wait(return_status_t)
  {
    return wait(arrive(), return_status);
  }

  _CCCL_HOST_DEVICE_API void arrive_and_wait(ignore_status_t)
  {
    wait(arrive(), ignore_status);
  }

  _CCCL_HOST_DEVICE_API void arrive_and_drop()
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80, (__arrive_and_drop(); return;))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status test_wait(int __phase, return_status_t) const
  {
#  if __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __make_operation_status(__test_wait_phase_status(__phase_value(__phase)));))
#  endif // __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return __test_wait_phase_completion_only(__phase);))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool test_wait(int __phase, ignore_status_t) const
  {
    const operation_status __status = test_wait(__phase, return_status);
    return __completed_ignoring_report(__status);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait(int __phase, return_status_t) const
  {
#  if __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __make_operation_status(__try_wait_phase_status(__phase_value(__phase)));))
#  endif // __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return __try_wait_phase_completion_only(__phase);))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait(int __phase, ignore_status_t) const
  {
    const operation_status __status = try_wait(__phase, return_status);
    return __completed_ignoring_report(__status);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status wait(int __phase, return_status_t) const
  {
    operation_status __result;
    do
    {
      __result = try_wait(__phase, return_status);
    } while (!__result.complete());
    return __result;
  }

  _CCCL_HOST_DEVICE_API void wait(int __phase, ignore_status_t) const
  {
    while (!try_wait(__phase, ignore_status))
    {
    }
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool test_wait_conditional_phase(int __phase) const
  {
#  if __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __test_wait_conditional_phase(__phase_value(__phase));))
#  endif // __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return __test_wait_phase_completion_only(__phase).complete();))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_conditional_phase(int __phase) const
  {
#  if __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __try_wait_conditional_phase(__phase_value(__phase));))
#  endif // __cccl_ptx_isa >= 940

    return test_wait_conditional_phase(__phase);
  }

  _CCCL_HOST_DEVICE_API void wait_conditional_phase(int __phase) const
  {
    while (!try_wait_conditional_phase(__phase))
    {
    }
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status
  try_wait_for(arrival_token __token, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, return_status_t) const
  {
    const auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    if (__nanosec.count() < 1)
    {
      return test_wait(__token, return_status);
    }

#  if __cccl_ptx_isa >= 940
    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (operation_status __result; const ::cuda::std::chrono::high_resolution_clock::time_point __start =
                                    ::cuda::std::chrono::high_resolution_clock::now();
       ::cuda::std::chrono::nanoseconds __elapsed(0);
       do {
         const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
         __result  = __make_operation_status(__try_wait_status(__token_value(__token), __wait_nsec));
         __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       } while (!__result.complete() && (__nanosec > __elapsed));
       return __result;))
#  endif // __cccl_ptx_isa >= 940
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __make_completion_only_operation_status(__try_wait_for(__token_value(__token), __nanosec));))

    __unsupported_storage();
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool
  try_wait_for(arrival_token __token, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, ignore_status_t) const
  {
    const operation_status __status = try_wait_for(__token, __dur, return_status);
    return __completed_ignoring_report(__status);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait_until(
    arrival_token __token, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time, return_status_t) const
  {
    return try_wait_for(__token, (__time - _Clock::now()), return_status);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_until(
    arrival_token __token, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time, ignore_status_t) const
  {
    return try_wait_for(__token, (__time - _Clock::now()), ignore_status);
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status
  try_wait_for(int __phase, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, return_status_t) const
  {
    const auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    if (__nanosec.count() < 1)
    {
      return test_wait(__phase, return_status);
    }

#  if __cccl_ptx_isa >= 940
    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (operation_status __result; const ::cuda::std::chrono::high_resolution_clock::time_point __start =
                                    ::cuda::std::chrono::high_resolution_clock::now();
       ::cuda::std::chrono::nanoseconds __elapsed(0);
       do {
         const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
         __result  = __make_operation_status(__try_wait_phase_status(__phase_value(__phase), __wait_nsec));
         __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       } while (!__result.complete() && (__nanosec > __elapsed));
       return __result;))
#  endif // __cccl_ptx_isa >= 940
    NV_IF_TARGET(
      NV_PROVIDES_SM_80,
      (return __make_completion_only_operation_status(__try_wait_phase_for(__phase_value(__phase), __nanosec));))

    __unsupported_storage();
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool
  try_wait_for(int __phase, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, ignore_status_t) const
  {
    const operation_status __status = try_wait_for(__phase, __dur, return_status);
    return __completed_ignoring_report(__status);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status
  try_wait_until(int __phase, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time, return_status_t) const
  {
    return try_wait_for(__phase, (__time - _Clock::now()), return_status);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool
  try_wait_until(int __phase, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time, ignore_status_t) const
  {
    return try_wait_for(__phase, (__time - _Clock::now()), ignore_status);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::ptrdiff_t max(shared_barrier_kind __kind) noexcept
  {
    return __max_for_kind(__kind);
  }
};

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint64_t* barrier_native_handle(::cuda::shared_barrier& __b)
{
  return __b.native_handle();
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 3)

#include <cuda/std/__cccl/epilogue.h>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 3)
#  include <cuda/__barrier/shared_barrier_tx.h>
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 3)

#endif // _CUDA___BARRIER_SHARED_BARRIER_H
