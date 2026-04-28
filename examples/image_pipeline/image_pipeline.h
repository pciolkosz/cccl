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

#ifndef IMAGE_PIPELINE_H
#define IMAGE_PIPELINE_H

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/stream>

// ── Constants ────────────────────────────────────────────────────────

/// Pixel type: 8-bit grayscale.
using pixel_t = cuda::std::uint8_t;

/// Image dimensions.  65K x 65K (~4 GB raw).  Large enough that
/// the working set won't fit in most GPUs, forcing real tiling.
/// Note: the full image is held in pinned host memory (~4 GB).
/// Ensure the system has at least 8 GB of RAM.
inline constexpr int image_width  = 65536;
inline constexpr int image_height = 65536;
inline constexpr cuda::std::size_t image_pixels = static_cast<cuda::std::size_t>(image_width) * image_height;

/// Preview downscale factor.  The output preview images are
/// (image_width / preview_scale) x (image_height / preview_scale).
inline constexpr int preview_scale = 64;

/// Histogram bins (one per possible grayscale value).
inline constexpr int num_bins   = cuda::std::numeric_limits<pixel_t>::max() + 1;
inline constexpr int num_levels = num_bins + 1; // CUB needs num_levels = num_bins + 1

// ── Shared data structures ───────────────────────────────────────────

/// Device selection and tile sizing results.
struct device_plan
{
  cuda::device_ref device;
  int tile_rows;
  int num_tiles;
  cuda::std::size_t gpu_budget;
};

/// All working memory for the pipeline.
struct tile_buffers
{
  cuda::device_buffer<pixel_t> dev_tile[2];      // double-buffered H2D
  cuda::device_buffer<float> dev_float_tile;     // normalized float tile
  cuda::device_buffer<int> dev_histogram;        // per-tile histogram
  cuda::device_buffer<float4> dev_tile_stats;    // per-tile reduction: {count, min, max, sum}
  cuda::device_buffer<pixel_t> dev_lut;          // equalization LUT
  cuda::device_buffer<pixel_t> dev_equalized;    // equalized tile
  cuda::host_buffer<pixel_t> host_image;         // full image in pinned memory
  cuda::host_buffer<int> host_tile_histograms;   // per-tile histograms
  cuda::host_buffer<float4> host_tile_stats;     // per-tile stats readback
  cuda::device_buffer<pixel_t> dev_preview;      // downscaled preview tile

  cuda::mr::shared_resource<cuda::device_memory_pool> device_pool;
  cuda::std::size_t tile_pixels;
};

/// Per-tile processing statistics.
struct tile_stats
{
  float min_val;
  float max_val;
  float sum;
  long long num_selected;
};

#endif // IMAGE_PIPELINE_H
