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

#include "cuda/experimental/async.cuh"

namespace
{
//! Scheduler that returns a sender that always completes inline
//! (successfully).
struct inline_scheduler
{
private:
  struct env_t
  {
    USTDEX_HOST_DEVICE auto query(cudax::get_completion_scheduler_t<cudax::set_value_t>) const noexcept
    {
      return inline_scheduler{};
    }
  };

  template <class Rcvr>
  struct opstate_t : cudax::_immovable
  {
    using operation_state_concept = cudax::operation_state_t;
    using completion_signatures   = cudax::completion_signatures<cudax::set_value_t()>;

    Rcvr _rcvr;

    USTDEX_HOST_DEVICE void start() noexcept
    {
      cudax::set_value(static_cast<Rcvr&&>(_rcvr));
    }
  };

  struct sndr_t
  {
    using sender_concept        = cudax::sender_t;
    using completion_signatures = cudax::completion_signatures<cudax::set_value_t()>;

    template <class Rcvr>
    USTDEX_HOST_DEVICE opstate_t<Rcvr> connect(Rcvr rcvr) const
    {
      return {{}, static_cast<Rcvr&&>(rcvr)};
    }

    USTDEX_HOST_DEVICE env_t get_env() const noexcept
    {
      return {};
    }
  };

  USTDEX_HOST_DEVICE friend bool operator==(inline_scheduler, inline_scheduler) noexcept
  {
    return true;
  }

  USTDEX_HOST_DEVICE friend bool operator!=(inline_scheduler, inline_scheduler) noexcept
  {
    return false;
  }

public:
  using scheduler_concept = cudax::scheduler_t;

  inline_scheduler() = default;

  USTDEX_HOST_DEVICE sndr_t schedule() const noexcept
  {
    return {};
  }
};
} // namespace
