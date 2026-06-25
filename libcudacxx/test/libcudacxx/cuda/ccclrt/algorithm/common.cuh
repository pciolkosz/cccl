//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LIBCUDACXX_CCCLRT_ALGORITHM_COMMON_CUH
#define TEST_LIBCUDACXX_CCCLRT_ALGORITHM_COMMON_CUH

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/mdspan>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr uint8_t fill_byte    = 1;
inline constexpr uint32_t buffer_size = 42;

inline int get_expected_value(uint8_t pattern_byte)
{
  int result;
  memset(&result, pattern_byte, sizeof(int));
  return result;
}

template <typename Result>
void check_result_and_erase(cuda::stream_ref stream, Result&& result, uint8_t pattern_byte = fill_byte)
{
  int expected = get_expected_value(pattern_byte);

  stream.sync();
  for (size_t i = 0; i < result.size(); ++i)
  {
    CCCLRT_REQUIRE(result.data()[i] == expected);
    result.data()[i] = 0;
  }
}

template <typename Layout = cuda::std::layout_right, typename Extents>
auto make_buffer_for_mdspan(cuda::stream_ref stream, Extents extents, char value = 0)
{
  auto mapping = typename Layout::template mapping<decltype(extents)>{extents};

  auto buffer = cuda::make_pinned_buffer<int>(stream, mapping.required_span_size(), cuda::no_init);

  memset(buffer.data(), value, buffer.size() * sizeof(int));

  return buffer;
}

inline auto create_fake_strided_mdspan()
{
  cuda::std::dextents<size_t, 3> dynamic_extents{1, 2, 3};
  cuda::std::array<size_t, 3> strides{12, 4, 1};
#if _CCCL_CUDACC_BELOW(12, 6)
  auto map = cuda::std::layout_stride::mapping{dynamic_extents, strides};
#else
  cuda::std::layout_stride::mapping map{dynamic_extents, strides};
#endif
  return cuda::std::mdspan<int, decltype(dynamic_extents), cuda::std::layout_stride>(nullptr, map);
};

#endif // TEST_LIBCUDACXX_CCCLRT_ALGORITHM_COMMON_CUH
