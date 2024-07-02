
#include <cuda/experimental/__async_alloc/async_alloc.cuh>
#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/memory_resource>

#include "../hierarchy/testing_common.cuh"

__global__ void kernel(cuda::std::span<int, 256> span, int val)
{
#if __CUDA_ARCH__ >= 700
  __nanosleep(10000000);
#endif
  span[2] = val;
}

TEST_CASE("Smoke", "[async_alloc]")
{
  cuda::experimental::mr::cuda_async_memory_resource res;
  cudaStream_t stream, different_stream;

  CUDART(cudaStreamCreate(&stream));
  CUDART(cudaStreamCreate(&different_stream));

  {
    auto box = cuda::experimental::alloc_async<int>(stream, 256, [](auto& async_buffer) {
      return cuda::std::span<int, 256>(async_buffer);
    });

    box.empty();

    auto dims = cudax::make_hierarchy(cudax::block_dims<256>(), cudax::grid_dims(2));
    cudax::launch(stream, dims, kernel, box, 42);

    cudax::launch(different_stream, dims, kernel, box, 42);
  }

  CUDART(cudaStreamSynchronize(different_stream));
  CUDART(cudaStreamSynchronize(stream));
}
