//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EVENT_DETAIL_H
#define __CUDAX_EVENT_DETAIL_H

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/stream_ref>

#include <cassert>

namespace cuda::experimental
{
/**
 * @brief An owning wrapper for a `cudaEvent_t`.
 */
struct event
{
  /**
   * @brief Flags to use when creating the event.
   */
  enum flags : unsigned int
  {
    default_       = cudaEventDefault,
    blocking_sync  = cudaEventBlockingSync,
    disable_timing = cudaEventDisableTiming,
    interprocess   = cudaEventInterprocess
  };

  /**
   * @brief Construct a new event object with the default flags.
   */
  event()
      : event(default_)
  {}

  /**
   * @brief Construct a new event object with the default flags.
   *
   * @throws cuda_error if the event creation fails.
   */
  explicit event(flags flags)
  {
    auto status = cudaEventCreateWithFlags(&event_, static_cast<unsigned int>(flags));
    if (status != cudaSuccess)
    {
      ::cuda::__throw_cuda_error(status, "Failed to create CUDA event");
    }
  }

  /**
   * @brief Construct a new event object into the moved-from state.
   *
   * @post `native_handle()` returns `cudaEvent_t()`.
   */
  explicit event(uninit_t) noexcept {}

  /**
   * @brief Move-construct a new event object
   *
   * @param other
   *
   * @post `other` is in a moved-from state.
   */
  event(event&& other) noexcept
      : event_(std::exchange(other.event_, {}))
  {
    other.event_ = nullptr;
  }

  /**
   * @brief Records an event on the specified stream
   *
   * @param stream
   *
   * @throws cuda_error if the event record fails
   */
  void record(stream_ref stream)
  {
    assert(event_ != nullptr);
    assert(stream.get() != nullptr);
    auto status = cudaEventRecord(event_, stream.get());
    if (status != cudaSuccess)
    {
      ::cuda::__throw_cuda_error(status, "Failed to record CUDA event");
    }
  }

  /**
   * @brief Waits for a CUDA event to complete on the specified stream
   *
   * @param stream The stream to wait on
   *
   * @throws cuda_error if the event wait fails
   */
  void wait(stream_ref stream) const
  {
    assert(event_ != nullptr);
    assert(stream.get() != nullptr);
    auto status = cudaStreamWaitEvent(stream.get(), event_);
    if (status != cudaSuccess)
    {
      ::cuda::__throw_cuda_error(status, "Failed to wait for CUDA event");
    }
  }

  /**
   * @brief Destroy the event object
   *
   * @note If the event fails to be destroyed, the error is silently ignored.
   */
  ~event()
  {
    if (event_ != nullptr)
    {
      [[maybe_unused]] auto status = cudaEventDestroy(event_);
    }
  }

  /**
   * @brief Construct an event object from a native `cudaEvent_t` handle.
   *
   * @param e The native handle
   *
   * @return event The constructed event object
   *
   * @note The constructed event object takes ownership of the native handle.
   */
  static event from_native_handle(cudaEvent_t e)
  {
    return event(e);
  }

  /**
   * @brief Retrieve the native `cudaEvent_t` handle.
   *
   * @return cudaEvent_t The native handle being held by the event object.
   */
  cudaEvent_t get() const noexcept
  {
    return event_;
  }

  /**
   * @brief Retrieve the native `cudaEvent_t` handle and give up ownership.
   *
   * @return cudaEvent_t The native handle being held by the event object.
   *
   * @post The event object is in a moved-from state.
   */
  cudaEvent_t release() noexcept
  {
    return std::exchange(event_, {});
  }

private:
  explicit event(cudaEvent_t event)
      : event_(event)
  {}

  friend flags operator|(flags lhs, flags rhs)
  {
    return static_cast<flags>(static_cast<unsigned int>(lhs) | static_cast<unsigned int>(rhs));
  }

  cudaEvent_t event_ = nullptr;
};
} // namespace cuda::experimental

#endif // __CUDAX_EVENT_DETAIL_H
