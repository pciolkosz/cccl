#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void sort_by_key(my_system& system, RandomAccessIterator1, RandomAccessIterator1, RandomAccessIterator2)
{
  system.validate_dispatch();
}

void TestSortByKeyDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::sort_by_key(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSortByKeyDispatchExplicit);

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void sort_by_key(my_tag, RandomAccessIterator1 keys_first, RandomAccessIterator1, RandomAccessIterator2)
{
  *keys_first = 13;
}

void TestSortByKeyDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::sort_by_key(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSortByKeyDispatchImplicit);

template <class Vector>
void InitializeSimpleKeyValueSortTest(
  Vector& unsorted_keys, Vector& unsorted_values, Vector& sorted_keys, Vector& sorted_values)
{
  unsorted_keys.resize(7);
  unsorted_keys = {1, 3, 6, 5, 2, 0, 4};
  unsorted_values.resize(7);
  unsorted_values = {0, 1, 2, 3, 4, 5, 6};

  sorted_keys.resize(7);
  sorted_keys = {0, 1, 2, 3, 4, 5, 6};
  sorted_values.resize(7);
  sorted_values = {5, 0, 4, 1, 6, 3, 2};
}

template <class Vector>
void TestSortByKeySimple()
{
  Vector unsorted_keys, unsorted_values;
  Vector sorted_keys, sorted_values;

  InitializeSimpleKeyValueSortTest(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

  thrust::sort_by_key(unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin());

  ASSERT_EQUAL(unsorted_keys, sorted_keys);
  ASSERT_EQUAL(unsorted_values, sorted_values);
}
DECLARE_VECTOR_UNITTEST(TestSortByKeySimple);

template <typename T>
void TestSortAscendingKeyValue(const size_t n)
{
  thrust::host_vector<T> h_keys   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_keys = h_keys;

  thrust::host_vector<T> h_values   = h_keys;
  thrust::device_vector<T> d_values = d_keys;

  thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), ::cuda::std::less<T>());
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), ::cuda::std::less<T>());

  ASSERT_EQUAL(h_keys, d_keys);
  ASSERT_EQUAL(h_values, d_values);
}
DECLARE_VARIABLE_UNITTEST(TestSortAscendingKeyValue);

template <typename T>
void TestSortDescendingKeyValue(const size_t n)
{
  thrust::host_vector<int> h_keys   = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_keys = h_keys;

  thrust::host_vector<int> h_values   = h_keys;
  thrust::device_vector<int> d_values = d_keys;

  thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), ::cuda::std::greater<int>());
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), ::cuda::std::greater<int>());

  ASSERT_EQUAL(h_keys, d_keys);
  ASSERT_EQUAL(h_values, d_values);
}
DECLARE_VARIABLE_UNITTEST(TestSortDescendingKeyValue);

void TestSortByKeyBool()
{
  const size_t n = 10027;

  thrust::host_vector<bool> h_keys  = unittest::random_integers<bool>(n);
  thrust::host_vector<int> h_values = unittest::random_integers<int>(n);

  thrust::device_vector<bool> d_keys  = h_keys;
  thrust::device_vector<int> d_values = h_values;

  thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

  ASSERT_EQUAL(h_keys, d_keys);
  ASSERT_EQUAL(h_values, d_values);
}
DECLARE_UNITTEST(TestSortByKeyBool);

void TestSortByKeyBoolDescending()
{
  const size_t n = 10027;

  thrust::host_vector<bool> h_keys  = unittest::random_integers<bool>(n);
  thrust::host_vector<int> h_values = unittest::random_integers<int>(n);

  thrust::device_vector<bool> d_keys  = h_keys;
  thrust::device_vector<int> d_values = h_values;

  thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), ::cuda::std::greater<bool>());
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), ::cuda::std::greater<bool>());

  ASSERT_EQUAL(h_keys, d_keys);
  ASSERT_EQUAL(h_values, d_values);
}
DECLARE_UNITTEST(TestSortByKeyBoolDescending);
