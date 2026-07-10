---
name: cccl-runtime
description: Use when writing, migrating, or reviewing CCCL C++ tests and benchmarks that should use CCCL Runtime APIs such as cuda::stream, cuda::event, cuda::buffer, cuda::launch, and Runtime memory resources instead of lower-level CUDA Runtime utilities.
---

# CCCL Runtime

## Source Of Truth

Use `docs/libcudacxx/runtime.rst` and the `docs/libcudacxx/runtime/` subpages as the authoritative Runtime API list and usage reference. Check the relevant headers under `libcudacxx/include/cuda/` when behavior or CTK availability matters.

## Migration Scope

- Work in semantic file families and keep batches small.
- Within selected files, apply all already-approved Runtime patterns comprehensively.
- Ask before introducing a new replacement category for the first time.
- Exclude examples and `cudax` unless the user explicitly changes the scope.

## Test Patterns

- Prefer `cuda::stream stream{device}` over raw `cudaStream_t`; pass `stream.get()` only to APIs that still require a native handle, such as `thrust::cuda::par.on(...)`.
- When a host-side test uses a stream policy, call `stream.sync()` before leaving the test body, even if the algorithm returns a scalar or iterator-like result.
- Prefer `cuda::make_device_buffer<T>(stream, device, ...)` and `cuda::buffer` over `thrust::device_vector` for migrated Runtime tests. Direct braced initializer lists are supported when constructing buffers.
- Avoid host dereference of device-buffer iterators. Compare iterator offsets, assert in a device functor, or copy results back with `cuda::copy_bytes` and synchronize the stream before host assertions.
- For fill/generate-style tests where all output elements should have one value, use a stream-ordered buffer check that copies to a host `std::vector<typename Buffer::value_type>` and compares each element. Do not reintroduce `thrust::host_vector` solely to build that reference.
- For host-side Runtime buffers in tests, use a synchronous adapter around `cuda::mr::legacy_pinned_memory_resource` when `cuda::make_pinned_buffer` would be gated by CTK availability. If the helper intentionally needs a stable device, use `cuda::device_ref{0}` explicitly. Use these host buffers for expected-result computation when migrating tests that previously used `thrust::host_vector`.
- When updating expected host Runtime buffers, using an established host-side Thrust algorithm can be a clear oracle and gives useful coverage for Runtime buffer host iterators. Prefer an independent loop only when the host algorithm would make the oracle too coupled to the behavior under test.
- Preserve tests whose purpose is raw pointer coverage, but use Runtime buffers as the backing allocation and synchronize the initialization stream before using a pointer with a default-stream Thrust policy.

## Benchmark Patterns

- In nvbench-based CUB benchmarks, use the benchmark stream as the ordering primitive: `cuda::stream_ref stream{state.get_cuda_stream().get_stream()}`.
- Derive the Runtime device from the stream with `stream.device()` when allocating setup buffers, so benchmark setup follows the device selected by nvbench.
- Pass the benchmark stream to CUB device APIs through an execution environment, for example `cuda::std::execution::env{stream}` or the local `cub_bench_env(...)` helper when an allocator is also needed.
- When constructing a CUB benchmark env outside a `state.exec` launch callback, combine the stream with the caching allocator explicitly: `cuda::std::execution::env{stream, cuda::std::execution::prop{cuda::mr::get_memory_resource, cuda::mr::resource_ref<>{alloc}}}`.
- For setup-only scalar outputs that kernels write and the host reads after `stream.sync()`, use a host-accessible, device-accessible Runtime buffer backed by `cuda::mr::synchronous_resource_adapter<cuda::mr::legacy_pinned_memory_resource>`.
- Keep shared data generators on their existing return type until the generator API itself is intentionally updated; use Runtime buffers for local benchmark setup storage that the benchmark owns directly.

## Launch Patterns

- Prefer `cuda::launch(stream, config, functor{}, args...)` over raw `<<<...>>>` launches.
- Use functors with `__device__ operator()` rather than templated `__global__` kernels when the call fits the Runtime launch model.
- Buffers passed to `cuda::launch` become `cuda::std::span` arguments; take the span in the functor and call `begin()` / `end()` inside the functor.
- If the launched algorithm needs the full output range size, use the span’s `size()` inside the functor instead of passing a duplicate size argument from the host.
- For subrange algorithms, pass the buffer plus small scalar offsets/counts into the launch functor, then compute `span.begin() + offset` inside the functor. Avoid passing host-computed begin/end iterators into `cuda::launch` when the whole buffer is already an argument.
- For launched algorithms that return an iterator into a passed buffer, prefer asserting the returned iterator against `span.begin() + expected_offset` inside the device functor instead of copying iterator objects back to the host.
- For device-side algorithms that return scalar results, pass the expected value into the launched functor and assert on device instead of allocating a one-element result buffer solely to copy the result back.
- If one functor would have predicate and non-predicate overloads where `cuda::launch` could treat the config as a first functor argument, split them into separate functor types to keep overload resolution unambiguous.
- After a launch that performs device-side assertions, call `stream.sync()` so failures are observed at the test site.

## Assertions

- Use the local test harness assertions on host.
- Use the Runtime test helper’s device assertion macro in launched functors when validating device-side algorithm results.
