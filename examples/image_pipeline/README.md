
# Image Processing Pipeline Example

A multi-file example showcasing the CCCL Runtime and CUB APIs working together
in a semi-realistic tiled image processing pipeline.

## What it does

The example generates a synthetic 65K x 65K (~4 GB) grayscale space observation
on the GPU, then processes it in tiles that fit in GPU memory:

1. **Pass 1 — Histogram**: Upload each tile, compute per-tile histograms with
   `cub::DeviceHistogram`, download and accumulate into a global histogram.

2. **Host interlude**: Compute Otsu's threshold (optimal foreground/background
   split) and build a histogram equalization lookup table from the CDF.

3. **Pass 2 — Equalize + Analyze**: For each tile, apply the equalization LUT
   (`cub::DeviceTransform`), normalize to float, compact bright pixels
   (`cub::DeviceSelect::If`), compute min/max/sum (`cub::DeviceReduce`), and
   GPU-downscale a preview.

4. **Output**: Write `input_preview.pgm` and `equalized_preview.pgm` (1024 x
   1024 previews). The equalized image reveals nebula structure and stars that
   are barely visible in the dark original.

## CCCL APIs demonstrated

| File | APIs |
|------|------|
| `device_setup.cu` | `cuda::devices`, `cuda::device_ref`, `cuda::device_attributes`, `cuda::arch_traits_for` |
| `tile_transfer.cu` | `cuda::device_memory_pool`, `cuda::memory_pool_properties`, `cuda::mr::shared_resource`, `cuda::make_pinned_buffer`, `cuda::make_buffer`, `cuda::no_init`, `cuda::copy_bytes`, `cuda::copy_configuration`, `cuda::fill_bytes`, `cuda::memory_pool_attributes`, `buffer.first()`, `buffer.subspan()` |
| `processing.cu` | CUB env-based APIs (`DeviceHistogram`, `DeviceTransform`, `DeviceSelect`, `DeviceReduce`), `cuda::std::execution::env`/`prop`, `cuda::mr::get_memory_resource_t`, `cuda::launch`, `cuda::distribute` |
| `generate_image.cu` | `cuda::launch`, `cuda::distribute`, `cuda::copy_bytes` |
| `main.cu` | `cuda::stream`, `cuda::event`, `cuda::timed_event`, `stream.wait(event)`, `stream.is_done()`, `buffer.get_unsynchronized()` |

## Building and running

```bash
cd examples/image_pipeline
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
./build/image_pipeline
```

The example requires ~4 GB of pinned host memory for the full image and uses
60% of GPU memory for the per-tile working set. It should run on any GPU with
at least 4 GB of memory.
