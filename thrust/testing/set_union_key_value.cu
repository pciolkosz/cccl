#include <thrust/functional.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename U>
void TestSetUnionKeyValue(size_t n)
{
  using T = key_value<U, U>;

  thrust::host_vector<U> h_keys_a   = unittest::random_integers<U>(n);
  thrust::host_vector<U> h_values_a = unittest::random_integers<U>(n);

  thrust::host_vector<U> h_keys_b   = unittest::random_integers<U>(n);
  thrust::host_vector<U> h_values_b = unittest::random_integers<U>(n);

  thrust::host_vector<T> h_a(n), h_b(n);
  for (size_t i = 0; i < n; ++i)
  {
    h_a[i] = T(h_keys_a[i], h_values_a[i]);
    h_b[i] = T(h_keys_b[i], h_values_b[i]);
  }

  thrust::stable_sort(h_a.begin(), h_a.end());
  thrust::stable_sort(h_b.begin(), h_b.end());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(h_a.size() + h_b.size());
  thrust::device_vector<T> d_result(d_a.size() + d_b.size());

  typename thrust::host_vector<T>::iterator h_end;
  typename thrust::device_vector<T>::iterator d_end;

  h_end = thrust::set_union(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
  h_result.erase(h_end, h_result.end());

  d_end = thrust::set_union(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());
  d_result.erase(d_end, d_result.end());

  ASSERT_EQUAL_QUIET(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestSetUnionKeyValue);

template <typename U>
void TestSetUnionKeyValueDescending(size_t n)
{
  using T = key_value<U, U>;

  thrust::host_vector<U> h_keys_a   = unittest::random_integers<U>(n);
  thrust::host_vector<U> h_values_a = unittest::random_integers<U>(n);

  thrust::host_vector<U> h_keys_b   = unittest::random_integers<U>(n);
  thrust::host_vector<U> h_values_b = unittest::random_integers<U>(n);

  thrust::host_vector<T> h_a(n), h_b(n);
  for (size_t i = 0; i < n; ++i)
  {
    h_a[i] = T(h_keys_a[i], h_values_a[i]);
    h_b[i] = T(h_keys_b[i], h_values_b[i]);
  }

  thrust::stable_sort(h_a.begin(), h_a.end(), ::cuda::std::greater<T>());
  thrust::stable_sort(h_b.begin(), h_b.end(), ::cuda::std::greater<T>());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(h_a.size() + h_b.size());
  thrust::device_vector<T> d_result(d_a.size() + d_b.size());

  typename thrust::host_vector<T>::iterator h_end;
  typename thrust::device_vector<T>::iterator d_end;

  h_end =
    thrust::set_union(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin(), ::cuda::std::greater<T>());
  h_result.erase(h_end, h_result.end());

  d_end =
    thrust::set_union(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin(), ::cuda::std::greater<T>());
  d_result.erase(d_end, d_result.end());

  ASSERT_EQUAL_QUIET(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestSetUnionKeyValueDescending);
