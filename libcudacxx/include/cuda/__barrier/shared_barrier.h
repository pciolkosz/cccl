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

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 4)
#  include <cuda/__barrier/shared_mbarrier.h>
#  include <cuda/__fwd/barrier.h>
#  include <cuda/__memory/address_space.h>
#  include <cuda/__utility/status_policy.h>
#  include <cuda/std/__atomic/scopes.h>
#  include <cuda/std/__chrono/duration.h>
#  include <cuda/std/__chrono/high_resolution_clock.h>
#  include <cuda/std/__chrono/time_point.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/terminate.h>
#  include <cuda/std/cstdint>

#  include <nv/target>

#  include <cuda_runtime_api.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint64_t* barrier_native_handle(::cuda::shared_barrier& __b);
_CCCL_END_NAMESPACE_CUDA_DEVICE

_CCCL_BEGIN_NAMESPACE_CUDA

class shared_barrier : private ::cuda::__detail::__shared_mbarrier_impl
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

    _CCCL_HOST_DEVICE_API constexpr operation_status(::cuda::__detail::__mbarrier_wait_status __result) noexcept
        : operation_status(__result.__complete, __result.__report_predicate, __result.__report_value)
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
        __report_inspected_ = true;
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
      __report_inspected_ = true;
      return __error_count(__source);
    }

    template <class _Fn>
    _CCCL_DEVICE_API void for_each_error(status_source __source, _Fn __fn) const noexcept
    {
      __report_inspected_        = true;
      const unsigned int __count = __error_count(__source);
      for (unsigned int __index = 0; __index != __count; ++__index)
      {
        // TODO: Consider wrapping cudaFabricOpStatusInfo before making this API public.
        __fn(__error_status(__source, __index));
      }
    }

    [[nodiscard]] _CCCL_DEVICE_API status_action classify(status_source __source) const noexcept
    {
      __report_inspected_ = true;
      _CCCL_ASSERT(__report_predicate_, "cannot classify a shared_barrier operation_status without a report");
      if (!__report_predicate_)
      {
        NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
        _CCCL_UNREACHABLE();
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
    _CCCL_HOST_DEVICE_API constexpr arrival_token() noexcept {}

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
    _CCCL_ASSERT(false, "shared_barrier requires local shared memory and mbarrier layout v1 support");
    NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
    _CCCL_UNREACHABLE();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::ptrdiff_t __max_expected_count() noexcept
  {
    return (1 << 9) - 1;
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

public:
  _CCCL_HOST_DEVICE_API ~shared_barrier()
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__inval(); return;))

    __unsupported_storage();
  }

  _CCCL_HOST_DEVICE_API inline friend void init(shared_barrier* __b, ::cuda::std::ptrdiff_t __expected)
  {
    _CCCL_ASSERT(1 <= __expected, "Expected arrival count must be at least one.");
    _CCCL_ASSERT(__expected <= shared_barrier::max(), "Expected arrival count cannot exceed shared_barrier::max().");

    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (__b->__assert_supported_storage(); __b->__init_status_reporting(static_cast<::cuda::std::uint32_t>(__expected));
       return;))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API arrival_token arrive(::cuda::std::ptrdiff_t __update = 1)
  {
    _CCCL_ASSERT(1 <= __update, "Arrival count update must be at least one.");
    _CCCL_ASSERT(__update <= shared_barrier::max(), "Arrival count update cannot exceed shared_barrier::max().");

    NV_IF_TARGET(NV_PROVIDES_SM_90, (return arrival_token(__arrive(__update));))

    __unsupported_storage();
  }

  _CCCL_HOST_DEVICE_API void expect_tx(::cuda::std::ptrdiff_t __transaction_count_update)
  {
    _CCCL_ASSERT(0 <= __transaction_count_update, "Transaction count update must be non-negative.");
    _CCCL_ASSERT(__transaction_count_update <= __max_transaction_count_update(),
                 "Transaction count update cannot exceed the mbarrier transaction count limit.");

    NV_IF_TARGET(NV_PROVIDES_SM_90, (__expect_tx(__transaction_count_update); return;))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API arrival_token
  arrive_tx(::cuda::std::ptrdiff_t __arrive_count_update, ::cuda::std::ptrdiff_t __transaction_count_update)
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

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status test_wait(arrival_token __token, return_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return operation_status(__test_wait_status(__token_value(__token)));))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool test_wait(arrival_token __token, ignore_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __test_wait(__token_value(__token));))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait(arrival_token __token, return_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return operation_status(__try_wait_status(__token_value(__token)));))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait(arrival_token __token, ignore_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __try_wait(__token_value(__token));))

    __unsupported_storage();
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
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__arrive_and_drop(); return;))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status test_wait(::cuda::std::uint32_t __phase, return_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return operation_status(__test_wait_phase_status(__phase));))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool test_wait(::cuda::std::uint32_t __phase, ignore_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __test_wait_phase(__phase);))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait(::cuda::std::uint32_t __phase, return_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return operation_status(__try_wait_phase_status(__phase));))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait(::cuda::std::uint32_t __phase, ignore_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __try_wait_phase(__phase);))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status wait(::cuda::std::uint32_t __phase, return_status_t) const
  {
    operation_status __result;
    do
    {
      __result = try_wait(__phase, return_status);
    } while (!__result.complete());
    return __result;
  }

  _CCCL_HOST_DEVICE_API void wait(::cuda::std::uint32_t __phase, ignore_status_t) const
  {
    while (!try_wait(__phase, ignore_status))
    {
    }
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool test_wait_conditional_phase(::cuda::std::uint32_t __phase) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __test_wait_conditional_phase(__phase);))

    __unsupported_storage();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_conditional_phase(::cuda::std::uint32_t __phase) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __try_wait_conditional_phase(__phase);))

    __unsupported_storage();
  }

  _CCCL_HOST_DEVICE_API void wait_conditional_phase(::cuda::std::uint32_t __phase) const
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

    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (operation_status __result; const ::cuda::std::chrono::high_resolution_clock::time_point __start =
                                    ::cuda::std::chrono::high_resolution_clock::now();
       ::cuda::std::chrono::nanoseconds __elapsed(0);
       do {
         const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
         __result  = operation_status(__try_wait_status(__token_value(__token), __wait_nsec));
         __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       } while (!__result.complete() && (__nanosec > __elapsed));
       return __result;))

    __unsupported_storage();
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool
  try_wait_for(arrival_token __token, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, ignore_status_t) const
  {
    const auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    if (__nanosec.count() < 1)
    {
      return test_wait(__token, ignore_status);
    }

    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (bool __complete = false; const ::cuda::std::chrono::high_resolution_clock::time_point __start =
                                  ::cuda::std::chrono::high_resolution_clock::now();
       ::cuda::std::chrono::nanoseconds __elapsed(0);
       do {
         const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
         __complete                              = __try_wait(__token_value(__token), __wait_nsec);
         __elapsed                               = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       } while (!__complete && (__nanosec > __elapsed));
       return __complete;))

    __unsupported_storage();
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
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait_for(
    ::cuda::std::uint32_t __phase, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, return_status_t) const
  {
    const auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    if (__nanosec.count() < 1)
    {
      return test_wait(__phase, return_status);
    }

    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (operation_status __result; const ::cuda::std::chrono::high_resolution_clock::time_point __start =
                                    ::cuda::std::chrono::high_resolution_clock::now();
       ::cuda::std::chrono::nanoseconds __elapsed(0);
       do {
         const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
         __result                                = operation_status(__try_wait_phase_status(__phase, __wait_nsec));
         __elapsed                               = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       } while (!__result.complete() && (__nanosec > __elapsed));
       return __result;))

    __unsupported_storage();
  }

  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_for(
    ::cuda::std::uint32_t __phase, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, ignore_status_t) const
  {
    const auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    if (__nanosec.count() < 1)
    {
      return test_wait(__phase, ignore_status);
    }

    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (bool __complete = false; const ::cuda::std::chrono::high_resolution_clock::time_point __start =
                                  ::cuda::std::chrono::high_resolution_clock::now();
       ::cuda::std::chrono::nanoseconds __elapsed(0);
       do {
         const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
         __complete                              = __try_wait_phase(__phase, __wait_nsec);
         __elapsed                               = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       } while (!__complete && (__nanosec > __elapsed));
       return __complete;))

    __unsupported_storage();
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait_until(
    ::cuda::std::uint32_t __phase,
    const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time,
    return_status_t) const
  {
    return try_wait_for(__phase, (__time - _Clock::now()), return_status);
  }

  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_until(
    ::cuda::std::uint32_t __phase,
    const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time,
    ignore_status_t) const
  {
    return try_wait_for(__phase, (__time - _Clock::now()), ignore_status);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::ptrdiff_t max() noexcept
  {
    return __max_expected_count();
  }
};

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint64_t* barrier_native_handle(::cuda::shared_barrier& __b)
{
  return __b.native_handle();
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 4)

#endif // _CUDA___BARRIER_SHARED_BARRIER_H
