//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#define LIBCUDACXX_ENABLE_EXCEPTIONS
#include <cuda/experimental/async.cuh>

#include "common/checked_receiver.hpp"
#include "common/error_scheduler.hpp"
#include "common/impulse_scheduler.hpp"
#include "common/stopped_scheduler.hpp"
#include "common/utility.hpp"

// Catch header in its own header block
#include "common/catch2.hpp" // IWYU pragma: keep

namespace
{
TEST_CASE("when_all simple example", "[when_all]")
{
  auto snd  = cudax::when_all(cudax::just(3), cudax::just(0.1415));
  auto snd1 = std::move(snd) | cudax::then([] USTDEX_HOST_DEVICE(int x, double y) {
                return x + y;
              });
  auto op   = cudax::connect(std::move(snd1), checked_value_receiver{3.1415});
  cudax::start(op);
}

TEST_CASE("when_all returning two values can be waited on", "[when_all]")
{
  auto snd = cudax::when_all(cudax::just(2), cudax::just(3));
  check_values(std::move(snd), 2, 3);
}

TEST_CASE("when_all with 5 senders", "[when_all]")
{
  auto snd = cudax::when_all(cudax::just(2), cudax::just(3), cudax::just(5), cudax::just(7), cudax::just(11));
  check_values(std::move(snd), 2, 3, 5, 7, 11);
}

TEST_CASE("when_all with just one sender", "[when_all]")
{
  auto snd = cudax::when_all(cudax::just(2));
  check_values(std::move(snd), 2);
}

TEST_CASE("when_all with move-only types", "[when_all]")
{
  auto snd = cudax::when_all(cudax::just(movable(2)));
  check_values(std::move(snd), movable(2));
}

TEST_CASE("when_all with no senders", "[when_all]")
{
  auto snd = cudax::when_all();
  check_values(std::move(snd));
}

TEST_CASE("when_all when one sender sends void", "[when_all]")
{
  auto snd = cudax::when_all(cudax::just(2), cudax::just());
  check_values(std::move(snd), 2);
}

#if USTDEX_HOST_ONLY()

TEST_CASE("when_all completes when children complete", "[when_all]")
{
  impulse_scheduler sched;
  bool called{false};
  auto snd = cudax::when_all(cudax::just(11) | cudax::continue_on(sched),
                             cudax::just(13) | cudax::continue_on(sched),
                             cudax::just(17) | cudax::continue_on(sched))
           | cudax::then([&](int a, int b, int c) {
               called = true;
               return a + b + c;
             });
  auto op = cudax::connect(std::move(snd), checked_value_receiver{41});
  cudax::start(op);
  // The when_all scheduler will complete only after 3 impulses
  CHECK_FALSE(called);
  sched.start_next();
  CHECK_FALSE(called);
  sched.start_next();
  CHECK_FALSE(called);
  sched.start_next();
  CHECK(called);
}

#endif

TEST_CASE("when_all can be used with just_*", "[when_all]")
{
  auto snd = cudax::when_all(cudax::just(2), cudax::just_error(42), cudax::just_stopped());
  auto op  = cudax::connect(std::move(snd), checked_error_receiver{42});
  cudax::start(op);
}

TEST_CASE("when_all terminates with error if one child terminates with error", "[when_all]")
{
  error_scheduler sched{42};
  auto snd = cudax::when_all(cudax::just(2), cudax::just(5) | cudax::continue_on(sched), cudax::just(7));
  auto op  = cudax::connect(std::move(snd), checked_error_receiver{42});
  cudax::start(op);
}

TEST_CASE("when_all terminates with stopped if one child is cancelled", "[when_all]")
{
  stopped_scheduler sched;
  auto snd = cudax::when_all(cudax::just(2), cudax::just(5) | cudax::continue_on(sched), cudax::just(7));
  auto op  = cudax::connect(std::move(snd), checked_stopped_receiver{});
  cudax::start(op);
}

#if USTDEX_HOST_ONLY()

TEST_CASE("when_all cancels remaining children if error is detected", "[when_all]")
{
  impulse_scheduler sched;
  error_scheduler err_sched{42};
  bool called1{false};
  bool called3{false};
  bool cancelled{false};
  auto snd = cudax::when_all(
    cudax::start_on(sched, cudax::just()) | cudax::then([&] {
      called1 = true;
    }),
    cudax::start_on(sched, cudax::just(5) | cudax::continue_on(err_sched)),
    cudax::start_on(sched, cudax::just()) | cudax::then([&] {
      called3 = true;
    }) | cudax::let_stopped([&] {
      cancelled = true;
      return cudax::just();
    }));
  auto op = cudax::connect(std::move(snd), checked_error_receiver{42});
  cudax::start(op);
  // The first child will complete; the third one will be cancelled
  CHECK_FALSE(called1);
  CHECK_FALSE(called3);
  sched.start_next(); // start the first child
  CHECK(called1);
  sched.start_next(); // start the second child; this will generate an error
  CHECK_FALSE(called3);
  sched.start_next(); // start the third child
  CHECK_FALSE(called3);
  CHECK(cancelled);
}

TEST_CASE("when_all cancels remaining children if cancel is detected", "[when_all]")
{
  stopped_scheduler stopped_sched;
  impulse_scheduler sched;
  bool called1{false};
  bool called3{false};
  bool cancelled{false};
  auto snd = cudax::when_all(
    cudax::start_on(sched, cudax::just()) | cudax::then([&] {
      called1 = true;
    }),
    cudax::start_on(sched, cudax::just(5) | cudax::continue_on(stopped_sched)),
    cudax::start_on(sched, cudax::just()) | cudax::then([&] {
      called3 = true;
    }) | cudax::let_stopped([&] {
      cancelled = true;
      return cudax::just();
    }));
  auto op = cudax::connect(std::move(snd), checked_stopped_receiver{});
  cudax::start(op);
  // The first child will complete; the third one will be cancelled
  CHECK_FALSE(called1);
  CHECK_FALSE(called3);
  sched.start_next(); // start the first child
  CHECK(called1);
  sched.start_next(); // start the second child; this will call set_stopped
  CHECK_FALSE(called3);
  sched.start_next(); // start the third child
  CHECK_FALSE(called3);
  CHECK(cancelled);
}

#endif

template <class... Ts>
struct just_ref
{
  using sender_concept        = cudax::sender_t;
  using completion_signatures = cudax::completion_signatures<cudax::set_value_t(Ts&...)>;
  USTDEX_HOST_DEVICE just_ref connect(cudax::_ignore) const
  {
    return {};
  }
};

TEST_CASE("when_all has the values_type based on the children, decayed and as rvalue "
          "references",
          "[when_all]")
{
  check_value_types<types<int>>(cudax::when_all(cudax::just(13)));
  check_value_types<types<double>>(cudax::when_all(cudax::just(3.14)));
  check_value_types<types<int, double>>(cudax::when_all(cudax::just(3, 0.14)));

  check_value_types<types<>>(cudax::when_all(cudax::just()));

  check_value_types<types<int, double>>(cudax::when_all(cudax::just(3), cudax::just(0.14)));
  check_value_types<types<int, double, int, double>>(
    cudax::when_all(cudax::just(3), cudax::just(0.14), cudax::just(1, 0.4142)));

  // if one child returns void, then the value is simply missing
  check_value_types<types<int, double>>(cudax::when_all(cudax::just(3), cudax::just(), cudax::just(0.14)));

  // if one child has no value completion, the when_all has no value
  // completion
  check_value_types<>(cudax::when_all(cudax::just(3), cudax::just_stopped(), cudax::just(0.14)));

  // if children send references, they get decayed
  check_value_types<types<int, double>>(cudax::when_all(just_ref<int>(), just_ref<double>()));
}

TEST_CASE("when_all has the error_types based on the children", "[when_all]")
{
  check_error_types<int>(cudax::when_all(cudax::just_error(13)));

  check_error_types<double>(cudax::when_all(cudax::just_error(3.14)));

  check_error_types<>(cudax::when_all(cudax::just()));

  check_error_types<int, double>(cudax::when_all(cudax::just_error(3), cudax::just_error(0.14)));

  check_error_types<int, double, string>(
    cudax::when_all(cudax::just_error(3), cudax::just_error(0.14), cudax::just_error(string{"err"})));

  check_error_types<error_code>(cudax::when_all(
    cudax::just(13), cudax::just_error(error_code{std::errc::invalid_argument}), cudax::just_stopped()));

#if USTDEX_HOST_ONLY()
  // if the child sends something with a potentially throwing decay-copy,
  // the when_all has an exception_ptr error completion.
  check_error_types<std::exception_ptr>(cudax::when_all(just_ref<potentially_throwing>()));
#else
  // in device code, there are no exceptions:
  check_error_types<>(cudax::when_all(just_ref<potentially_throwing>()));
#endif
}

TEST_CASE("when_all has the sends_stopped == true", "[when_all]")
{
  check_sends_stopped<true>(cudax::when_all(cudax::just(13)));
  check_sends_stopped<true>(cudax::when_all(cudax::just_error(-1)));
  check_sends_stopped<true>(cudax::when_all(cudax::just_stopped()));

  check_sends_stopped<true>(cudax::when_all(cudax::just(3), cudax::just(0.14)));
  check_sends_stopped<true>(cudax::when_all(cudax::just(3), cudax::just_error(-1), cudax::just_stopped()));
}
} // namespace
