#include <cub/cub.cuh>

void b()
{
  printf("b() called\n");

  cub::DoubleBuffer<unsigned int> d_keys;
  cub::DoubleBuffer<cub::NullType> d_values;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d_keys, d_values, 1024);
}
