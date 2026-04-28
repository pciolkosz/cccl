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

#ifndef DETAIL_H
#define DETAIL_H

/// @file
/// Supporting details for the image pipeline example: synthetic image
/// generation and printing/output helpers.  These are not part of the
/// pipeline itself — in a real application, image generation would be
/// replaced by actual data loading, and the printing would be replaced
/// by your application's logging.

#include "image_pipeline.h"

// ── Image generation ─────────────────────────────────────────────────

/// Generate a synthetic space observation on the GPU, tile by tile,
/// and produce a downscaled input preview.
/// In a real application this would be replaced by loading actual data.
void generate_image(cuda::stream_ref stream, tile_buffers& bufs, int num_tiles, cuda::std::span<pixel_t> host_preview);

// ── Printing / output helpers ────────────────────────────────────────

void print_device_info(cuda::device_ref dev, cuda::arch_traits_t traits, cuda::std::size_t total_mem);
void print_tile_plan(int tile_rows, int tile_alignment, int num_tiles, cuda::std::size_t budget, cuda::std::size_t total_mem);
void print_allocation_info(cuda::std::size_t device_total, cuda::std::size_t gpu_budget, cuda::std::size_t tile_pixels, int tile_rows);
void print_pool_stats(tile_buffers& bufs, cuda::device_ref device);

struct iqr_result
{
  int p25, p75;
  int span() const
  {
    return p75 - p25;
  }
};

iqr_result compute_iqr(cuda::std::span<const int> hist, cuda::std::size_t total);
void print_pass_stats(double ms, long long total_selected, double mean_selected, float global_min, float global_max);
void print_sanity_check(iqr_result orig, iqr_result eq);
void print_summary(int num_tiles, int tile_rows, double pass1_ms, double pass2_ms, bool ok);
void write_pgm(const char* filename, cuda::std::span<const pixel_t> data, int width, int height);

/// Downscale a device tile using CUB BlockReduce and copy the result
/// to the host preview buffer.
void downscale_tile(
  cuda::stream_ref stream,
  tile_buffers& bufs,
  cuda::std::span<const pixel_t> dev_src,
  int row_offset,
  int tile_rows,
  cuda::std::span<pixel_t> host_preview);

#endif // DETAIL_H
