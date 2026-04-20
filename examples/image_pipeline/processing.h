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

#ifndef PROCESSING_H
#define PROCESSING_H

#include <cuda/std/span>
#include <cuda/stream>

#include "tile_transfer.h"

struct tile_stats
{
  float min_val;
  float max_val;
  float sum;
  int num_selected;
};

/// Compute the histogram of pixel intensities for the current tile.
void compute_histogram(cuda::stream_ref stream, tile_buffers& bufs, int slot, size_t tile_pixel_count);

/// Given the global histogram, compute Otsu's threshold (minimizes
/// intra-class variance) and an equalization LUT that flattens the
/// intensity distribution.
float compute_otsu_threshold(cuda::std::span<const int> histogram, size_t total_pixels);

/// Build an equalization lookup table from a histogram.  Each input
/// intensity is mapped to an output that makes the CDF approximately
/// uniform.  The LUT is written to `lut_out[0..255]`.
void build_equalization_lut(cuda::std::span<const int> histogram, size_t total_pixels, cuda::std::span<pixel_t> lut_out);

/// Run the full processing chain on one tile:
///   1. Apply the equalization LUT          (DeviceTransform)
///   2. Normalize equalized uint8 → float   (DeviceTransform)
///   3. Histogram the equalized pixels      (DeviceHistogram)
///   4. Threshold + compact bright pixels   (DeviceSelect::If)
///   5. Reduce: min, max, sum               (DeviceReduce)
tile_stats process_tile(cuda::stream_ref stream, tile_buffers& bufs, int slot, size_t tile_pixel_count, float threshold);

/// Downscale the tile in dev_equalized on the GPU and copy the small
/// result to the host preview buffer.
void downscale_tile(cuda::stream_ref stream,
                    tile_buffers& bufs,
                    int tile_row_offset,
                    int tile_rows,
                    cuda::std::span<pixel_t> host_preview);

#endif // PROCESSING_H
