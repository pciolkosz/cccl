#include <thrust/detail/util/align.h>

#include <cuda_runtime_api.h>

#include <unittest/unittest.h>

template <typename T>
void TestCudaMallocResultAligned(const std::size_t n)
{
  T* ptr = 0;
  cudaMalloc(&ptr, n * sizeof(T));
  cudaFree(ptr);

  ASSERT_EQUAL(true, thrust::detail::util::is_aligned(ptr));
}
DECLARE_VARIABLE_UNITTEST(TestCudaMallocResultAligned);
