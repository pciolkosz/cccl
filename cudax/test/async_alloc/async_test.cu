
#include <cuda/experimental/__async_alloc/async_alloc.cuh>
#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/memory_resource>

#include "../hierarchy/testing_common.cuh"

__global__ void kernel(cuda::std::span<int, 256> span)
{
  span[2] = 1;
}

TEST_CASE("Smoke", "[async_alloc]")
{
  cuda::experimental::mr::cuda_async_memory_resource res;
  cudaStream_t stream;

  CUDART(cudaStreamCreate(&stream));

  auto buff = cuda::experimental::uninitialized_async_buffer<int>(res, stream, 256);

  auto box = cuda::experimental::alloc_async<int>(stream, 256, [](auto& async_buffer) {
    return cuda::std::span<int, 256>(async_buffer.data(), 256);
  });

  auto dims = cudax::make_hierarchy(cudax::block_dims<256>(), cudax::grid_dims(2));
  cudax::launch(stream, dims, kernel, box);

  CUDART(cudaStreamSynchronize(stream));
}
