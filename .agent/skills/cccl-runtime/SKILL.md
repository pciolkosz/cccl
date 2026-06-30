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
- Prefer `cuda::make_device_buffer<T>(stream, device, ...)` and `cuda::buffer` over `thrust::device_vector` for migrated Runtime tests. Direct braced initializer lists are supported when constructing buffers.
- Avoid host dereference of device-buffer iterators. Compare iterator offsets, assert in a device functor, or copy results back with `cuda::copy_bytes` and synchronize the stream before host assertions.
- For host-side Runtime buffers in tests, use a synchronous adapter around `cuda::mr::legacy_pinned_memory_resource` when `cuda::make_pinned_buffer` would be gated by CTK availability. If the helper intentionally needs a stable device, use `cuda::device_ref{0}` explicitly.
- Preserve tests whose purpose is raw pointer coverage, but use Runtime buffers as the backing allocation and synchronize the initialization stream before using a pointer with a default-stream Thrust policy.

## Launch Patterns

- Prefer `cuda::launch(stream, config, functor{}, args...)` over raw `<<<...>>>` launches.
- Use functors with `__device__ operator()` rather than templated `__global__` kernels when the call fits the Runtime launch model.
- Buffers passed to `cuda::launch` become `cuda::std::span` arguments; take the span in the functor and call `begin()` / `end()` inside the functor.
- If one functor would have predicate and non-predicate overloads where `cuda::launch` could treat the config as a first functor argument, split them into separate functor types to keep overload resolution unambiguous.
- After a launch that performs device-side assertions, call `stream.sync()` so failures are observed at the test site.

## Assertions

- Use the local test harness assertions on host.
- Use the Runtime test helper’s device assertion macro in launched functors when validating device-side algorithm results.
