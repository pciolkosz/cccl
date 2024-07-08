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

#ifndef USTDEX_NAMESPACE
#  define USTDEX_NAMESPACE cuda::experimental
#endif

// Include this first
#include "__async/config.hpp"

// Include the other implementation headers:
#include "__async/basic_sender.hpp"
#include "__async/conditional.hpp"
#include "__async/continue_on.hpp"
#include "__async/cpos.hpp"
#include "__async/just.hpp"
#include "__async/just_from.hpp"
#include "__async/let_value.hpp"
#include "__async/queries.hpp"
#include "__async/read_env.hpp"
#include "__async/run_loop.hpp"
#include "__async/sequence.hpp"
#include "__async/start_detached.hpp"
#include "__async/start_on.hpp"
#include "__async/stop_token.hpp"
#include "__async/sync_wait.hpp"
#include "__async/then.hpp"
#include "__async/thread_context.hpp"
#include "__async/when_all.hpp"
#include "__async/write_env.hpp"
