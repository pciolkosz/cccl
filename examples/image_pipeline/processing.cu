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
 * Processing — CUB-based image analysis pipeline.
 *
 * Demonstrates:
 *   cub::DeviceTransform         apply equalization LUT and normalize
 *   cub::DeviceHistogram         compute intensity histogram
 *   cub::DeviceSelect::If        compact pixels above a threshold
 *   cub::DeviceReduce            min, max, sum of selected pixels
 *   CUB env-based APIs           pass stream + memory resource via env,
 *                                so CUB allocates temporaries from our pool
 *   cuda::std::execution::env    compose stream and memory resource into env
 *   cuda::std::execution::prop   bind a query to a value in an env
 *   cuda::copy_bytes             readback reduction results to host
 *
 * All CUB calls use the environment-based overloads.  We build an env
 * containing both the stream and a reference to the device pool, so CUB
 * allocates its temporary storage from our pool instead of the default.
 *
 * Otsu's method and histogram equalization are host-side algorithms that
 * depend on the histogram readback — creating a natural sync point
 * between pass 1 and pass 2.
 */

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/memory_resource>
#include <cuda/std/execution>
#include <cuda/std/span>

#include <cmath>

#include "image_pipeline.h"
#include "processing.h"

// ── Build a CUB execution env with stream + our memory pool ──────────
// CUB's env-based APIs query cuda::get_stream and cuda::mr::get_memory_resource
// from the env to determine which stream to launch on and where to allocate
// temporary storage.
static auto make_cub_env(cuda::stream_ref stream, cuda::mr::shared_resource<cuda::device_memory_pool>& pool)
{
  auto mr_prop = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, pool};
  return cuda::std::execution::env{stream, mr_prop};
}

// ── Apply equalization LUT: input pixel → LUT[pixel] ─────────────────
struct equalize_op
{
  const pixel_t* lut;
  __device__ pixel_t operator()(pixel_t p) const
  {
    return lut[p];
  }
};

// ── Normalize: uint8 → float [0, 1] ─────────────────────────────────
struct normalize_op
{
  __device__ float operator()(pixel_t p) const
  {
    return static_cast<float>(p) / 255.0f;
  }
};

// ── Threshold predicate ──────────────────────────────────────────────
struct above_threshold
{
  float thresh;
  __device__ bool operator()(float v) const
  {
    return v > thresh;
  }
};

// ── Sum functor for DeviceReduce ─────────────────────────────────────
struct sum_op
{
  __device__ float operator()(float a, float b) const
  {
    return a + b;
  }
};

void compute_histogram(cuda::stream_ref stream, tile_buffers& bufs, int slot, size_t tile_pixel_count)
{
  auto env = make_cub_env(stream, bufs.device_pool);
  (void) cub::DeviceHistogram::HistogramEven(
    bufs.dev_tile[slot].first(tile_pixel_count).data(),
    bufs.dev_histogram.data(),
    num_levels,
    0,
    256,
    static_cast<int>(tile_pixel_count),
    env);
}

tile_stats process_tile(cuda::stream_ref stream, tile_buffers& bufs, int slot, size_t tile_pixel_count, float threshold)
{
  int n = static_cast<int>(tile_pixel_count);

  auto pixel_data = bufs.dev_tile[slot].first(tile_pixel_count).data();
  auto eq_data    = bufs.dev_equalized.data();
  auto float_data = bufs.dev_float_tile.data();
  auto env        = make_cub_env(stream, bufs.device_pool);

  // ── 1. Apply equalization LUT ──────────────────────────────────────
  // Each pixel is remapped through the LUT to flatten the intensity
  // distribution.  The LUT was uploaded before this pass.
  (void) cub::DeviceTransform::Transform(pixel_data, eq_data, n, equalize_op{bufs.dev_lut.data()}, env);

  // ── 2. Normalize equalized uint8 → float ───────────────────────────
  (void) cub::DeviceTransform::Transform(eq_data, float_data, n, normalize_op{}, env);

  // ── 3. Histogram of equalized pixels ───────────────────────────────
  // This histogram goes into the same device buffer — we download and
  // accumulate it on the host after this tile.
  (void) cub::DeviceHistogram::HistogramEven(eq_data, bufs.dev_histogram.data(), num_levels, 0, 256, n, env);

  // ── 4. Threshold + compact ─────────────────────────────────────────
  (void) cub::DeviceSelect::If(
    float_data,
    bufs.dev_compact.data(),
    bufs.dev_num_selected.data(),
    static_cast<::cuda::std::int64_t>(n),
    above_threshold{threshold},
    env);

  // ── Read back num_selected ─────────────────────────────────────────
  cuda::copy_bytes(stream, bufs.dev_num_selected, bufs.host_num_selected);
  stream.sync();

  int num_selected = bufs.host_num_selected.get_unsynchronized(0);

  tile_stats stats{};
  stats.num_selected = num_selected;

  if (num_selected == 0)
  {
    return stats;
  }

  // ── 5. Reduce: min, max, sum of compacted pixels ───────────────────
  // Each reduction writes to a different element of dev_reduce_out so
  // all three can be in-flight on the same stream without intermediate
  // syncs.  One copy_bytes + one sync at the end.
  (void) cub::DeviceReduce::Min(bufs.dev_compact.data(), bufs.dev_reduce_out.data() + 0, num_selected, env);
  (void) cub::DeviceReduce::Max(bufs.dev_compact.data(), bufs.dev_reduce_out.data() + 1, num_selected, env);
  (void) cub::DeviceReduce::Reduce(
    bufs.dev_compact.data(), bufs.dev_reduce_out.data() + 2, num_selected, sum_op{}, 0.0f, env);

  cuda::copy_bytes(stream, bufs.dev_reduce_out, bufs.host_reduce_out);
  stream.sync();

  stats.min_val = bufs.host_reduce_out.get_unsynchronized(0);
  stats.max_val = bufs.host_reduce_out.get_unsynchronized(1);
  stats.sum     = bufs.host_reduce_out.get_unsynchronized(2);

  return stats;
}

// ── Otsu's method ────────────────────────────────────────────────────
float compute_otsu_threshold(cuda::std::span<const int> histogram, size_t total_pixels)
{
  double total_sum = 0;
  for (int i = 0; i < num_bins; ++i)
  {
    total_sum += static_cast<double>(i) * histogram[i];
  }

  double sum_bg    = 0;
  double weight_bg = 0;
  double max_var   = 0;
  int best_t       = 0;

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

    double var = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
    if (var > max_var)
    {
      max_var = var;
      best_t  = t;
    }
  }

  return static_cast<float>(best_t) / 255.0f;
}

// ── Downscale kernel ─────────────────────────────────────────────────
struct downscale_kernel
{
  template <typename Config>
  __device__ void
  operator()(Config config, cuda::std::span<const pixel_t> src, cuda::std::span<pixel_t> dst, int src_width, int scale)
  {
    auto tid      = cuda::gpu_thread.rank(cuda::grid, config);
    int dst_width = src_width / scale;
    if (tid >= dst.size())
    {
      return;
    }
    int px  = static_cast<int>(tid) % dst_width;
    int py  = static_cast<int>(tid) / dst_width;
    int sum = 0;
    for (int dy = 0; dy < scale; ++dy)
    {
      for (int dx = 0; dx < scale; ++dx)
      {
        sum += src[static_cast<size_t>(py * scale + dy) * src_width + (px * scale + dx)];
      }
    }
    dst[tid] = static_cast<pixel_t>(sum / (scale * scale));
  }
};

void downscale_tile(
  cuda::stream_ref stream, tile_buffers& bufs, int tile_row_offset, int tile_rows, cuda::std::span<pixel_t> host_preview)
{
  int dst_rows   = tile_rows / preview_scale;
  int dst_cols   = image_width / preview_scale;
  int dst_pixels = dst_rows * dst_cols;
  if (dst_pixels == 0)
  {
    return;
  }

  constexpr int block_size = 256;
  auto config              = cuda::distribute<block_size>(dst_pixels);
  size_t src_pixels        = static_cast<size_t>(tile_rows) * image_width;
  cuda::launch(
    stream,
    config,
    downscale_kernel{},
    bufs.dev_equalized.first(src_pixels),
    bufs.dev_preview.first(static_cast<size_t>(dst_pixels)),
    image_width,
    preview_scale);

  int preview_row_offset = tile_row_offset / preview_scale;
  cuda::copy_bytes(
    stream,
    bufs.dev_preview.first(static_cast<size_t>(dst_pixels)),
    host_preview.subspan(static_cast<size_t>(preview_row_offset) * dst_cols, static_cast<size_t>(dst_pixels)));
}

// ── Histogram equalization LUT ───────────────────────────────────────
// Compute the CDF from the histogram, normalize it to [0, 255], and
// write the result as a byte-valued lookup table.
void build_equalization_lut(cuda::std::span<const int> histogram, size_t total_pixels, cuda::std::span<pixel_t> lut_out)
{
  double scale = 255.0 / static_cast<double>(total_pixels);
  double cdf   = 0;
  for (int i = 0; i < num_bins; ++i)
  {
    cdf += histogram[i];
    int mapped = static_cast<int>(cuda::std::min(cdf * scale, 255.0));
    lut_out[i] = static_cast<pixel_t>(mapped);
  }
}
