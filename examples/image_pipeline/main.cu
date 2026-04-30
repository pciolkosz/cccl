/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Image processing pipeline — all runtime API usage in one file.
 *
 * This file contains the complete pipeline:
 *   - Device selection and tile sizing
 *   - Memory pool creation and buffer allocation
 *   - Tile upload/download with copy_bytes and fill_bytes
 *   - CUB-based processing (histogram, equalization, thresholding, reduction)
 *   - GPU downscale with CUB BlockReduce
 *   - Double-buffered two-stream orchestration
 *
 * Supporting details (image generation, printing) are in detail.cu.
 */

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/stream>

#include <cstdio>

#include "detail.h"
#include "image_pipeline.h"

// ═════════════════════════════════════════════════════════════════════
// Device selection
// ═════════════════════════════════════════════════════════════════════

static device_plan select_device_and_plan()
{
  printf("=== Device selection ===\n");

  cuda::device_ref best      = cuda::devices[0];
  cuda::std::size_t best_mem = 0;

  for (auto dev : cuda::devices)
  {
    cuda::std::size_t total_bytes = dev.attribute(cuda::device_attributes::total_global_memory);
    int sms                       = dev.attribute(cuda::device_attributes::multiprocessor_count);
    auto name                     = dev.name();
    printf("  [%d] %.*s  %3d SMs  %.0f MB\n",
           dev.get(),
           static_cast<int>(name.size()),
           name.data(),
           sms,
           total_bytes / (1024.0 * 1024.0));

    if (total_bytes > best_mem)
    {
      best     = dev;
      best_mem = total_bytes;
    }
  }

  auto cc     = best.attribute(cuda::device_attributes::compute_capability);
  auto traits = cuda::arch_traits_for(cc);
  print_device_info(best, traits, best_mem);

  // Budget 60% of total GPU memory for the per-tile working set.
  cuda::std::size_t budget          = static_cast<cuda::std::size_t>(best_mem * 0.60);
  cuda::std::size_t overhead        = 128 * 1024 * 1024;
  cuda::std::size_t bytes_per_pixel = 2 * sizeof(pixel_t) + sizeof(pixel_t) + 2 * sizeof(float);
  cuda::std::size_t max_tile_pixels = (budget - overhead) / bytes_per_pixel;
  int tile_rows                     = static_cast<int>(max_tile_pixels / image_width);

  constexpr int tile_alignment = 32;
  tile_rows                    = (tile_rows / tile_alignment) * tile_alignment;
  tile_rows                    = cuda::std::clamp(tile_rows, tile_alignment, image_height);
  int num_tiles                = cuda::ceil_div(image_height, tile_rows);

  print_tile_plan(tile_rows, tile_alignment, num_tiles, budget, best_mem);
  return {best, tile_rows, num_tiles, budget};
}

// ═════════════════════════════════════════════════════════════════════
// Buffer allocation
// ═════════════════════════════════════════════════════════════════════

static tile_buffers allocate_tile_buffers(
  cuda::stream_ref stream, cuda::device_ref device, int tile_rows, cuda::std::size_t gpu_budget, int num_tiles)
{
  printf("=== Tile buffer allocation ===\n");
  cuda::std::size_t tile_pixels = static_cast<cuda::std::size_t>(tile_rows) * image_width;

  // Single pool for our buffers and CUB temporaries.
  cuda::std::size_t device_total =
    2 * tile_pixels * sizeof(pixel_t) // double-buffered pixel tiles
    + tile_pixels * sizeof(pixel_t) // equalized tile
    + tile_pixels * sizeof(float) // normalized float tile
    + sizeof(float4) // reduction output
    + num_bins * sizeof(pixel_t) // equalization LUT
    + num_bins * sizeof(int); // histogram

  cuda::memory_pool_properties props{};
  props.initial_pool_size = device_total;
  props.max_pool_size     = gpu_budget;

  auto device_pool = cuda::mr::shared_resource<cuda::device_memory_pool>(
    cuda::std::in_place_type<cuda::device_memory_pool>, device, props);

  // Device buffers — no_init since kernels/copies write before reading.
  auto dev_tile_0              = cuda::make_buffer<pixel_t>(stream, device_pool, tile_pixels, cuda::no_init);
  auto dev_tile_1              = cuda::make_buffer<pixel_t>(stream, device_pool, tile_pixels, cuda::no_init);
  auto dev_float               = cuda::make_buffer<float>(stream, device_pool, tile_pixels, cuda::no_init);
  auto dev_hist                = cuda::make_buffer<int>(stream, device_pool, num_bins, cuda::no_init);
  auto dev_tile_stats          = cuda::make_buffer<float4>(stream, device_pool, num_tiles, cuda::no_init);
  auto dev_lut                 = cuda::make_buffer<pixel_t>(stream, device_pool, num_bins, cuda::no_init);
  auto dev_equalized           = cuda::make_buffer<pixel_t>(stream, device_pool, tile_pixels, cuda::no_init);
  cuda::std::size_t preview_px = (tile_rows / preview_scale) * (image_width / preview_scale);
  auto dev_preview             = cuda::make_buffer<pixel_t>(stream, device_pool, preview_px, cuda::no_init);

  // Pinned host buffers — make_pinned_buffer uses the default pinned pool.
  auto host_image = cuda::make_pinned_buffer<pixel_t>(stream, image_pixels, pixel_t{0});
  auto host_tile_hists =
    cuda::make_pinned_buffer<int>(stream, static_cast<cuda::std::size_t>(num_tiles) * num_bins, int{0});
  auto host_tile_stats = cuda::make_pinned_buffer<float4>(stream, num_tiles, float4{0, 0, 0, 0});

  print_allocation_info(device_total, gpu_budget, tile_pixels, tile_rows);

  return {
    {cuda::std::move(dev_tile_0), cuda::std::move(dev_tile_1)},
    cuda::std::move(dev_float),
    cuda::std::move(dev_hist),
    cuda::std::move(dev_tile_stats),
    cuda::std::move(dev_lut),
    cuda::std::move(dev_equalized),
    cuda::std::move(host_image),
    cuda::std::move(host_tile_hists),
    cuda::std::move(host_tile_stats),
    cuda::std::move(dev_preview),
    device_pool,
    tile_pixels,
  };
}

// ═════════════════════════════════════════════════════════════════════
// Tile transfer helpers
// ═════════════════════════════════════════════════════════════════════

static cuda::std::size_t upload_tile(cuda::stream_ref stream, tile_buffers& bufs, int slot, int tile_idx, int tile_rows)
{
  cuda::std::size_t offset = static_cast<cuda::std::size_t>(tile_idx) * bufs.tile_pixels;
  cuda::std::size_t count  = cuda::std::min(bufs.tile_pixels, image_pixels - offset);

  cuda::copy_configuration config{};
  config.src_access_order = cuda::source_access_order::stream;
  cuda::copy_bytes(stream, bufs.host_image.subspan(offset, count), bufs.dev_tile[slot].first(count), config);
  cuda::fill_bytes(stream, bufs.dev_histogram, cuda::std::uint8_t{0});
  return count;
}

static void download_tile_histogram(cuda::stream_ref stream, tile_buffers& bufs, int tile_idx)
{
  cuda::std::size_t offset = static_cast<cuda::std::size_t>(tile_idx) * num_bins;
  cuda::copy_bytes(stream, bufs.dev_histogram, bufs.host_tile_histograms.subspan(offset, num_bins));
}

static void accumulate_histograms(tile_buffers& bufs, int num_tiles, cuda::std::span<int> result)
{
  for (int i = 0; i < num_bins; ++i)
  {
    result[i] = 0;
  }
  for (int t = 0; t < num_tiles; ++t)
  {
    cuda::std::size_t offset = static_cast<cuda::std::size_t>(t) * num_bins;
    for (int i = 0; i < num_bins; ++i)
    {
      result[i] += bufs.host_tile_histograms.get_unsynchronized(offset + i);
    }
  }
}

static void upload_lut(cuda::stream_ref stream, tile_buffers& bufs, cuda::std::span<const pixel_t> host_lut)
{
  cuda::copy_bytes(stream, host_lut, bufs.dev_lut);
}

// ═════════════════════════════════════════════════════════════════════
// CUB-based processing
// ═════════════════════════════════════════════════════════════════════

static auto make_cub_env(cuda::stream_ref stream, cuda::mr::shared_resource<cuda::device_memory_pool>& pool)
{
  auto mr_prop = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, pool};
  return cuda::std::execution::env{stream, mr_prop};
}

static void check_cub(cudaError_t err, const char* msg)
{
  if (err != cudaSuccess)
  {
    fprintf(stderr, "CUB error in %s: %s\n", msg, cudaGetErrorString(err));
  }
}

// ── Kernel functors ──────────────────────────────────────────────────

struct equalize_op
{
  cuda::std::span<const pixel_t> lut;
  __device__ pixel_t operator()(pixel_t p) const
  {
    return lut[p];
  }
};

struct normalize_op
{
  __device__ float operator()(pixel_t p) const
  {
    return static_cast<float>(p) / cuda::std::numeric_limits<pixel_t>::max();
  }
};

// Combined count/min/max/sum reduction using float4: x=count, y=min, z=max, w=sum.
struct stats_reduce_op
{
  __device__ float4 operator()(float4 a, float4 b) const
  {
    return {a.x + b.x, cuda::std::min(a.y, b.y), cuda::std::max(a.z, b.z), a.w + b.w};
  }
};

// Transform op that maps each pixel to stats if above threshold, or identity if below.
struct threshold_stats_op
{
  float thresh;
  __device__ float4 operator()(float v) const
  {
    if (v > thresh)
    {
      return {1.0f, v, v, v}; // count=1, min=v, max=v, sum=v
    }
    return {0.0f, cuda::std::numeric_limits<float>::max(), cuda::std::numeric_limits<float>::lowest(), 0.0f};
  }
};

// ── Processing functions ─────────────────────────────────────────────

static void compute_histogram(cuda::stream_ref stream, tile_buffers& bufs, int slot, cuda::std::size_t tile_pixel_count)
{
  auto env = make_cub_env(stream, bufs.device_pool);
  check_cub(
    cub::DeviceHistogram::HistogramEven(
      bufs.dev_tile[slot].first(tile_pixel_count).data(),
      bufs.dev_histogram.data(),
      num_levels,
      0,
      num_bins,
      static_cast<int>(tile_pixel_count),
      env),
    "HistogramEven (pass 1)");
}

static void process_tile(
  cuda::stream_ref stream,
  tile_buffers& bufs,
  int slot,
  int tile_idx,
  cuda::std::size_t tile_pixel_count,
  float threshold)
{
  int n           = static_cast<int>(tile_pixel_count);
  auto pixel_data = bufs.dev_tile[slot].first(tile_pixel_count).data();
  auto eq_data    = bufs.dev_equalized.data();
  auto float_data = bufs.dev_float_tile.data();
  auto env        = make_cub_env(stream, bufs.device_pool);

  check_cub(cub::DeviceTransform::Transform(pixel_data, eq_data, n, equalize_op{bufs.dev_lut.first(num_bins)}, env),
            "Transform (equalize)");
  check_cub(cub::DeviceTransform::Transform(eq_data, float_data, n, normalize_op{}, env), "Transform (normalize)");
  check_cub(cub::DeviceHistogram::HistogramEven(eq_data, bufs.dev_histogram.data(), num_levels, 0, num_bins, n, env),
            "HistogramEven (pass 2)");

  // Combined threshold + count/min/max/sum in a single pass.
  // Each tile writes to its own slot — no sync needed between tiles.
  float4 identity{0.0f, cuda::std::numeric_limits<float>::max(), cuda::std::numeric_limits<float>::lowest(), 0.0f};

  check_cub(
    cub::DeviceReduce::TransformReduce(
      float_data,
      bufs.dev_tile_stats.data() + tile_idx,
      n,
      stats_reduce_op{},
      threshold_stats_op{threshold},
      identity,
      env),
    "TransformReduce");

  // D2H copy into this tile's slot — no sync, read after all tiles finish.
  cuda::copy_bytes(stream, bufs.dev_tile_stats.subspan(tile_idx, 1), bufs.host_tile_stats.subspan(tile_idx, 1));
}

/// Accumulate per-tile stats into a single result after all tiles finish.
static tile_stats accumulate_tile_stats(tile_buffers& bufs, int num_tiles)
{
  tile_stats result{};
  result.min_val = cuda::std::numeric_limits<float>::max();
  result.max_val = cuda::std::numeric_limits<float>::lowest();

  for (int t = 0; t < num_tiles; ++t)
  {
    auto s = bufs.host_tile_stats.get_unsynchronized(t);
    result.num_selected += static_cast<long long>(s.x);
    result.min_val = cuda::std::min(result.min_val, s.y);
    result.max_val = cuda::std::max(result.max_val, s.z);
    result.sum += s.w;
  }
  return result;
}

// ── Host-side algorithms ─────────────────────────────────────────────

static float compute_otsu_threshold(cuda::std::span<const int> histogram, cuda::std::size_t total_pixels)
{
  double total_sum = 0;
  for (int i = 0; i < num_bins; ++i)
  {
    total_sum += static_cast<double>(i) * histogram[i];
  }
  double sum_bg = 0, weight_bg = 0, max_var = 0;
  int best_t = 0;
  for (int t = 0; t < num_bins; ++t)
  {
    weight_bg += histogram[t];
    if (weight_bg == 0)
    {
      continue;
    }
    double weight_fg = static_cast<double>(total_pixels) - weight_bg;
    if (weight_fg == 0)
    {
      break;
    }
    sum_bg += static_cast<double>(t) * histogram[t];
    double mean_bg = sum_bg / weight_bg;
    double mean_fg = (total_sum - sum_bg) / weight_fg;
    double var     = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
    if (var > max_var)
    {
      max_var = var;
      best_t  = t;
    }
  }
  return static_cast<float>(best_t) / cuda::std::numeric_limits<pixel_t>::max();
}

static void build_equalization_lut(
  cuda::std::span<const int> histogram, cuda::std::size_t total_pixels, cuda::std::span<pixel_t> lut_out)
{
  constexpr double max_val = cuda::std::numeric_limits<pixel_t>::max();
  double scale             = max_val / static_cast<double>(total_pixels);
  double cdf               = 0;
  for (int i = 0; i < num_bins; ++i)
  {
    cdf += histogram[i];
    lut_out[i] = static_cast<pixel_t>(cuda::std::min(cdf * scale, max_val));
  }
}

// ═════════════════════════════════════════════════════════════════════
// Downscale kernel
// ═════════════════════════════════════════════════════════════════════

// Each block produces one output pixel by box-averaging a scale×scale
// source block.  Threads cooperatively load source pixels and sum
// them locally, then cub::BlockReduce merges the partial sums.
//
// The block size is extracted from the launch config at compile time
// via cuda::gpu_thread.count(cuda::block, config), which is then used
// as the template parameter for cub::BlockReduce.

struct downscale_kernel
{
  template <typename Config>
  __device__ void
  operator()(Config config, cuda::std::span<const pixel_t> src, cuda::std::span<pixel_t> dst, int src_width, int scale)
  {
    constexpr int block_size = cuda::gpu_thread.count(cuda::block, config);
    int out_idx              = blockIdx.x;
    if (out_idx >= static_cast<int>(dst.size()))
    {
      return;
    }

    int dst_width   = src_width / scale;
    int px          = out_idx % dst_width;
    int py          = out_idx / dst_width;
    int total_elems = scale * scale;

    // Each thread sums its share of the scale×scale source block.
    int local_sum = 0;
    int tid       = threadIdx.x;
    for (int i = tid; i < total_elems; i += block_size)
    {
      int dy = i / scale;
      int dx = i % scale;
      local_sum += src[static_cast<cuda::std::size_t>(py * scale + dy) * src_width + (px * scale + dx)];
    }

    // cub::BlockReduce sums the per-thread partial sums into a single
    // block-wide total.  Without CUB, this would be a manual shared-
    // memory tree reduction:
    //
    //   __shared__ int smem[block_size];
    //   smem[tid] = local_sum;
    //   __syncthreads();
    //   for (int s = block_size / 2; s > 0; s >>= 1)
    //   {
    //     if (tid < s) smem[tid] += smem[tid + s];
    //     __syncthreads();
    //   }
    //   int block_sum = smem[0];
    using BlockReduceT = cub::BlockReduce<int, block_size>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    int block_sum = BlockReduceT(temp_storage).Sum(local_sum);

    if (tid == 0)
    {
      dst[out_idx] = static_cast<pixel_t>(block_sum / total_elems);
    }
  }
};

void downscale_tile(
  cuda::stream_ref stream,
  tile_buffers& bufs,
  cuda::std::span<const pixel_t> dev_src,
  int row_offset,
  int tile_rows,
  cuda::std::span<pixel_t> host_preview)
{
  int dst_rows   = tile_rows / preview_scale;
  int dst_cols   = image_width / preview_scale;
  int dst_pixels = dst_rows * dst_cols;
  if (dst_pixels == 0)
  {
    return;
  }

  constexpr int block_size = 256;
  auto config              = cuda::make_config(cuda::block_dims<block_size>(), cuda::grid_dims(dst_pixels));

  cuda::launch(
    stream,
    config,
    downscale_kernel{},
    dev_src,
    bufs.dev_preview.first(static_cast<cuda::std::size_t>(dst_pixels)),
    image_width,
    preview_scale);

  int preview_row_offset = row_offset / preview_scale;
  cuda::copy_bytes(stream,
                   bufs.dev_preview.first(static_cast<cuda::std::size_t>(dst_pixels)),
                   host_preview.subspan(static_cast<cuda::std::size_t>(preview_row_offset) * dst_cols,
                                        static_cast<cuda::std::size_t>(dst_pixels)));
}

// ═════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════

int main()
{
  // ── 1. Device selection and tile sizing ────────────────────────────
  auto plan = select_device_and_plan();

  // ── 2. Allocate all buffers ────────────────────────────────────────
  // Two streams for double-buffered tile processing.  stream_a also
  // handles setup work (allocation, LUT upload, etc.) between passes.
  cuda::stream stream_a{plan.device};
  cuda::stream stream_b{plan.device};
  cuda::stream_ref streams[2] = {stream_a, stream_b};

  auto bufs = allocate_tile_buffers(stream_a, plan.device, plan.tile_rows, plan.gpu_budget, plan.num_tiles);

  // ── 3. Generate image and downscale input preview ──────────────────
  int pw = image_width / preview_scale;
  int ph = image_height / preview_scale;
  auto host_input_preview =
    cuda::make_pinned_buffer<pixel_t>(stream_a, static_cast<cuda::std::size_t>(pw) * ph, pixel_t{0});
  generate_image(stream_a, bufs, plan.num_tiles, host_input_preview.subspan(0));
  write_pgm("input_preview.pgm", host_input_preview.subspan(0), pw, ph);

  // ── 4. Pass 1: histogram (double-buffered) ─────────────────────────
  //   stream_a: [upload tile 0] [histogram 0] [download 0]       [tile 2] ...
  //   stream_b:                 [upload tile 1] [histogram 1] [download 1] ...
  printf("=== Pass 1: histogram ===\n");

  cuda::timed_event pass1_start{stream_a};

  for (int t = 0; t < plan.num_tiles; ++t)
  {
    int slot                = t % 2;
    cuda::std::size_t count = upload_tile(streams[slot], bufs, slot, t, plan.tile_rows);
    compute_histogram(streams[slot], bufs, slot, count);
    download_tile_histogram(streams[slot], bufs, t);
  }

  stream_a.wait(stream_b);
  cuda::timed_event pass1_end{stream_a};
  stream_a.sync();
  double pass1_ms = (pass1_end - pass1_start).count() / 1e6;
  printf("  Histogram pass: %.1f ms\n", pass1_ms);

  // ── 5. Otsu threshold + equalization LUT ───────────────────────────
  cuda::std::array<int, num_bins> original_hist{};
  cuda::std::span global_hist_span{original_hist};
  accumulate_histograms(bufs, plan.num_tiles, global_hist_span);

  float otsu = compute_otsu_threshold(global_hist_span, image_pixels);
  printf("  Otsu threshold: %.4f (%d / 255)\n", otsu, static_cast<int>(otsu * 255));

  pixel_t host_lut[num_bins];
  build_equalization_lut(global_hist_span, image_pixels, cuda::std::span<pixel_t>(host_lut, num_bins));

  // Construct a pinned buffer from the host LUT array — the buffer
  // copies the data in stream order, no manual sync needed.
  auto pinned_lut = cuda::make_pinned_buffer<pixel_t>(stream_a, host_lut, host_lut + num_bins);
  upload_lut(stream_a, bufs, pinned_lut.subspan(0));
  printf("  Equalization LUT uploaded\n\n");

  cuda::fill_bytes(stream_a, bufs.host_tile_histograms, cuda::std::uint8_t{0});
  auto host_eq_preview =
    cuda::make_pinned_buffer<pixel_t>(stream_a, static_cast<cuda::std::size_t>(pw) * ph, pixel_t{0});

  // stream_b must wait for the setup work on stream_a before starting pass 2.
  stream_b.wait(stream_a);

  // ── 6. Pass 2: equalize + threshold + stats + preview ──────────────
  printf("=== Pass 2: equalize + threshold + statistics ===\n");
  cuda::timed_event pass2_start{stream_a};

  for (int t = 0; t < plan.num_tiles; ++t)
  {
    int slot                = t % 2;
    cuda::std::size_t count = upload_tile(streams[slot], bufs, slot, t, plan.tile_rows);
    process_tile(streams[slot], bufs, slot, t, count, otsu);
    int tile_rows  = static_cast<int>(count / image_width);
    int row_offset = t * static_cast<int>(bufs.tile_pixels / image_width);

    downscale_tile(
      streams[slot], bufs, bufs.dev_equalized.first(count), row_offset, tile_rows, host_eq_preview.subspan(0));
    download_tile_histogram(streams[slot], bufs, t);
  }

  // Single sync after all tiles — stats, histograms, and preview are on host.
  stream_a.wait(stream_b);
  cuda::timed_event pass2_end{stream_a};
  stream_a.sync();
  double pass2_ms = (pass2_end - pass2_start).count() / 1e6;

  auto stats           = accumulate_tile_stats(bufs, plan.num_tiles);
  double mean_selected = (stats.num_selected > 0) ? static_cast<double>(stats.sum) / stats.num_selected : 0.0;

  print_pass_stats(pass2_ms, stats.num_selected, mean_selected, stats.min_val, stats.max_val);
  print_pool_stats(bufs, plan.device);

  // ── 7. Write equalized preview ─────────────────────────────────────
  write_pgm("equalized_preview.pgm", host_eq_preview.subspan(0), pw, ph);
  printf("\n");

  // ── 8. Sanity check ────────────────────────────────────────────────
  auto orig_iqr = compute_iqr(cuda::std::span{original_hist}, image_pixels);

  cuda::std::array<int, num_bins> equalized_hist{};
  accumulate_histograms(bufs, plan.num_tiles, cuda::std::span{equalized_hist});
  auto eq_iqr = compute_iqr(cuda::std::span{equalized_hist}, image_pixels);

  print_sanity_check(orig_iqr, eq_iqr);

  bool ok = eq_iqr.width() > orig_iqr.width();
  print_summary(plan.num_tiles, plan.tile_rows, pass1_ms, pass2_ms, ok);
  return ok ? 0 : 1;
}
