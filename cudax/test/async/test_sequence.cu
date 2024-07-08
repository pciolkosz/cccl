/*
 * Copyright (c) 2024 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Include this first
#include <cuda/experimental/async.cuh>

// Then include the test helpers
#include "common/checked_receiver.hpp"
#include "common/utility.hpp"

// Catch header in its own header block
#include "common/catch2.hpp" // IWYU pragma: keep

namespace
{
TEST_CASE("simple use of sequence executes both child operations", "[adaptors][sequence]")
{
  bool flag1{false};
  bool flag2{false};

  auto sndr1 = cudax::sequence(
    cudax::just() | cudax::then([&] {
      flag1 = true;
    }),
    cudax::just() | cudax::then([&] {
      flag2 = true;
    }));

  check_value_types<types<>>(sndr1);
#if USTDEX_HOST_ONLY()
  check_error_types<std::exception_ptr>(sndr1);
#else
  check_error_types<>(sndr1);
#endif
  check_sends_stopped<false>(sndr1);

  auto op = cudax::connect(std::move(sndr1), checked_value_receiver<>{});
  cudax::start(op);

  CHECK(flag1);
  CHECK(flag2);
}

} // namespace
