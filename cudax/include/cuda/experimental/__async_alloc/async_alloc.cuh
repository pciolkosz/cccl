#ifndef ASYNC_ALLOC
#define ASYNC_ALLOC
#include <cuda/experimental/__async_alloc/uninitialized_async_buffer>
#include <cuda/experimental/event.cuh>
#include <cuda/experimental/memory_resource>

#include <cassert>
#include <stdexcept>

namespace cuda::experimental
{

template <typename Container, typename T, typename... Properties>
struct async_allocation_box
{
  Container container;
  cuda::experimental::uninitialized_async_buffer<T, Properties...> buffer;
  cuda::experimental::event event;

  async_allocation_box(
    Container&& c, cuda::experimental::uninitialized_async_buffer<T, Properties...>&& b, cuda::experimental::event e)
      : container(std::move(c))
      , buffer(std::move(b))
      , event(std::move(e))
  {}
};

template <typename T, cuda::std::size_t Extent, typename... Properties>
struct async_allocation_box<cuda::std::span<T, Extent>, T, Properties...>
{
  cuda::std::span<T, Extent> container;
  cuda::experimental::uninitialized_async_buffer<T, Properties...> buffer;
  cuda::experimental::event event;

  using element_type = T;
  using size_type    = cuda::std::size_t;

  async_allocation_box(cuda::std::span<T, Extent>&& c,
                       cuda::experimental::uninitialized_async_buffer<T, Properties...>&& b,
                       cuda::experimental::event e)
      : container(std::move(c))
      , buffer(std::move(b))
      , event(std::move(e))
  {}

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
  if (box.buffer.stream() != stream)
  {
    box.event.wait(stream);
  }
  return box.container;
}

// This probably shouldn't be a separate overload, but otherwise not_box case is called
template <typename Container, typename T, typename... Properties>
auto& unpack_box_and_sync(async_allocation_box<Container, T, Properties...>& box, ::cuda::stream_ref stream)
{
  if (box.buffer.stream() != stream)
  {
    box.event.wait(stream);
  }
  return box.container;
}

template <typename Container, typename T, typename... Properties>
auto&& unpack_box_and_sync(async_allocation_box<Container, T, Properties...>&& box, ::cuda::stream_ref stream)
{
  if (box.buffer.stream() != stream)
  {
    box.event.wait(stream);
  }
  return std::move(box.container);
}

template <typename T>
auto&& unpack_box_and_sync(T&& not_box, ::cuda::stream_ref stream)
{
  return cuda::std::forward<T>(not_box);
}

template <typename Container, typename T, typename... Properties>
void box_wait_for_kernel(const async_allocation_box<Container, T, Properties...>& box, ::cuda::stream_ref stream)
{
  if (box.buffer.stream() != stream)
  {
    event e;
    e.record(stream);
    e.wait(box.buffer.stream());
  }
}

template <typename T>
void box_wait_for_kernel(const T& not_box, ::cuda::stream_ref stream)
{}
} // namespace detail

template <typename T, typename ResRef, typename Fn>
auto alloc_async(::cuda::stream_ref stream, ResRef res, size_t size, const Fn& fn)
{
  auto event = experimental::event(event::disable_timing);
  auto buff  = cuda::experimental::uninitialized_async_buffer<T>(res, stream, size);
  event.record(stream);
  auto container = fn(buff);

  return async_allocation_box{cuda::std::move(container), cuda::std::move(buff), std::move(event)};
}

template <typename T, typename Fn>
auto alloc_async(::cuda::stream_ref stream, size_t size, const Fn& fn)
{
  cuda::experimental::mr::cuda_async_memory_resource res;
  return alloc_async<T>(stream, res, size, fn);
}
} // namespace cuda::experimental
#endif
