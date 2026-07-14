//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "jit_dispatch_nvrtc_kernel.cuh"

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/experimental/kernel.cuh>
#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace example = cudax_example::jit_dispatch_nvrtc;
namespace cudax   = cuda::experimental;

namespace
{
cuda::experimental::prototype::jit_kernel_id select_copy_implementation(int count)
{
  if (count <= 32)
  {
    return example::small_copy_impl::id;
  }
  if (count % 4 == 0)
  {
    return example::unrolled_copy_4_impl::id;
  }
  if (count % 2 == 0)
  {
    return example::unrolled_copy_2_impl::id;
  }
  return example::scalar_copy_impl::id;
}

int parse_count(int argc, char** argv)
{
  if (argc <= 1)
  {
    return 1024;
  }

  const int count = std::atoi(argv[1]);
  if (count <= 0)
  {
    throw std::runtime_error("copy count must be positive");
  }
  return count;
}

void jit_copy(cudax::stream& stream, int* dst, const int* src, int count)
{
  const auto selected_id = select_copy_implementation(count);
  const auto dispatcher  = example::dispatcher_type::make_dispatcher(selected_id, dst, src, count);
  constexpr int block_size = 128;
  const int blocks         = (count + block_size - 1) / block_size;
  const auto launch_config = cuda::block_dims<block_size>() & cuda::grid_dims(blocks);

#if defined(CUDAX_EXAMPLE_JIT_DISPATCH_USE_NVRTC)
  cuda::experimental::prototype::jit_launch(stream, launch_config, dispatcher);
#else
  cudax::launch(stream, launch_config, dispatcher);
#endif
  stream.sync();
}
} // namespace

int main(int argc, char** argv)
try
{
  cuda::device_ref device{0};
  cudax::stream stream{device};

  auto managed_resource = cuda::managed_default_memory_pool();
  const int count       = parse_count(argc, argv);
  auto input            = cuda::make_buffer<int>(stream, managed_resource, count, cuda::no_init);
  auto output           = cuda::make_buffer<int>(stream, managed_resource, count, cuda::no_init);

  for (int i = 0; i != count; ++i)
  {
    input.get_unsynchronized(i)  = 17 + i;
    output.get_unsynchronized(i) = -1;
  }

  jit_copy(stream, output.data(), input.data(), count);
  for (int i = 0; i != count; ++i)
  {
    if (output.get_unsynchronized(i) != input.get_unsynchronized(i))
    {
      throw std::runtime_error("jit_copy produced an unexpected result");
    }
  }

  std::printf("jit_dispatch copy example passed\n");
  return EXIT_SUCCESS;
}
catch (const std::exception& error)
{
  std::fprintf(stderr, "%s\n", error.what());
  return EXIT_FAILURE;
}
catch (...)
{
  std::fprintf(stderr, "caught an unknown exception\n");
  return EXIT_FAILURE;
}
