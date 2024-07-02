#ifndef ASYNC_ALLOC
#define ASYNC_ALLOC
#include <cuda/experimental/__async_alloc/uninitialized_async_buffer>
#include <cuda/experimental/memory_resource>

namespace cuda::experimental
{

template <typename Container, typename T, typename... Properties>
struct async_allocation_box
{
  Container container;
  cuda::experimental::uninitialized_async_buffer<T, Properties...> buffer;
  cudaEvent_t event;

  async_allocation_box(
    Container&& c, cuda::experimental::uninitialized_async_buffer<T, Properties...>&& b, cudaEvent_t e)
      : container(std::move(c))
      , buffer(std::move(b))
      , event(e)
  {}

  ~async_allocation_box()
  {
    cudaEventDestroy(event);
  }
};

template <typename T, cuda::std::size_t Extent, typename... Properties>
struct async_allocation_box<cuda::std::span<T, Extent>, T, Properties...>
{
  cuda::std::span<T, Extent> container;
  cuda::experimental::uninitialized_async_buffer<T, Properties...> buffer;
  cudaEvent_t event;

  using element_type = T;
  using size_type    = cuda::std::size_t;

  async_allocation_box(
    cuda::std::span<T, Extent>&& c, cuda::experimental::uninitialized_async_buffer<T, Properties...>&& b, cudaEvent_t e)
      : container(std::move(c))
      , buffer(std::move(b))
      , event(e)
  {}

  ~async_allocation_box()
  {
    cudaEventDestroy(event);
  }

  size_type size() const
  {
    return container.size();
  }

  bool empty() const
  {
    return container.empty();
  }
};

namespace detail
{
template <typename Container, typename T, typename... Properties>
auto& unpack_box_and_sync(const async_allocation_box<Container, T, Properties...>& box, ::cuda::stream_ref stream)
{
  cudaStreamWaitEvent(stream.get(), box.event);
  return box.container;
}

template <typename Container, typename T, typename... Properties>
auto&& unpack_box_and_sync(async_allocation_box<Container, T, Properties...>&& box, ::cuda::stream_ref stream)
{
  cudaStreamWaitEvent(stream.get(), box.event);
  return std::move(box.container);
}

template <typename T>
auto& unpack_box_and_sync(const T& not_box, ::cuda::stream_ref stream)
{
  return not_box;
}

template <typename T>
auto&& unpack_box_and_sync(T&& not_box, ::cuda::stream_ref stream)
{
  return std::move(not_box);
}
} // namespace detail

template <typename T, typename ResRef, typename Fn>
auto alloc_async(::cuda::stream_ref stream, ResRef res, size_t size, const Fn& fn)
{
  cudaEvent_t event;

  cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  auto buff = cuda::experimental::uninitialized_async_buffer<T>(res, stream, size);

  auto container = fn(buff);

  return async_allocation_box{cuda::std::move(container), cuda::std::move(buff), event};
}

template <typename T, typename Fn>
auto alloc_async(::cuda::stream_ref stream, size_t size, const Fn& fn)
{
  cuda::experimental::mr::cuda_async_memory_resource res;

  return alloc_async<T>(stream, res, size, fn);
}
} // namespace cuda::experimental
#endif
