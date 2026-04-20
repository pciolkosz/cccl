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

#ifndef TILE_TRANSFER_H
#define TILE_TRANSFER_H

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/span>
#include <cuda/stream>

#include "image_pipeline.h"

/// Working memory for the tile transfer pipeline.
struct tile_buffers
{
  // Two device pixel tiles for double-buffered H2D upload.
  cuda::device_buffer<pixel_t> dev_tile[2];

  // Normalized float tile (processing works in float).
  cuda::device_buffer<float> dev_float_tile;

  // Compacted output (worst case: same size as float tile).
  cuda::device_buffer<float> dev_compact;

  // Number of selected pixels (device-side output of DeviceSelect).
  cuda::device_buffer<int> dev_num_selected;

  // Per-tile histogram bins (device side).
  cuda::device_buffer<int> dev_histogram;

  // Reduction output buffers (3 elements: min, max, sum) so all three
  // reductions can be in-flight without intermediate syncs.
  cuda::device_buffer<float> dev_reduce_out;

  // Equalization lookup table: maps input intensity [0..255] → output [0..255].
  // Uploaded once after pass 1, used by pass 2's normalization step.
  cuda::device_buffer<pixel_t> dev_lut;

  // Equalized output tile (device side).
  cuda::device_buffer<pixel_t> dev_equalized;

  // Full image in pinned host memory (input).
  cuda::host_buffer<pixel_t> host_image;

  // Per-tile histograms in pinned host memory.  Each tile downloads its
  // histogram into its own slot (num_bins ints), so all tiles can run
  // without synchronizing for accumulation.  Accumulated on the host
  // after all tiles finish.
  cuda::host_buffer<int> host_tile_histograms; // num_tiles * num_bins entries

  // Per-tile pixel count (how many survived thresholding).
  cuda::host_buffer<int> host_num_selected;

  // Host-side reduction readback.
  cuda::host_buffer<float> host_reduce_out;

  // Downscaled preview tile (device side).
  cuda::device_buffer<pixel_t> dev_preview;

  // The device memory pool shared across our buffers and CUB temporaries.
  cuda::mr::shared_resource<cuda::device_memory_pool> device_pool;

  size_t tile_pixels; // pixels per full tile
};

/// Allocate all buffers for the pipeline.
/// @param gpu_budget  Maximum bytes the device pool is allowed to use.
/// @param num_tiles   Total number of tiles (used to size per-tile histograms).
tile_buffers allocate_tile_buffers(
  cuda::stream_ref stream, cuda::device_ref device, int tile_rows, size_t gpu_budget, int num_tiles);

/// Upload tile `tile_idx` from host_image to device, using the double-
/// buffered slot `slot`.  Returns the pixel count for this tile (the
/// last tile may be smaller).
size_t upload_tile(cuda::stream_ref stream, tile_buffers& bufs, int slot, int tile_idx, int tile_rows);

/// Download the per-tile histogram from device to the tile's slot in
/// host_tile_histograms.  No sync needed — each tile has its own slot.
void download_tile_histogram(cuda::stream_ref stream, tile_buffers& bufs, int tile_idx);

/// Accumulate all per-tile histograms into a single result.
/// Call after all tiles have finished (both streams synced).
/// @param result  Output span of num_bins ints.
void accumulate_histograms(tile_buffers& bufs, int num_tiles, cuda::std::span<int> result);

/// Upload the equalization LUT from a host-side array to device.
void upload_lut(cuda::stream_ref stream, tile_buffers& bufs, cuda::std::span<const pixel_t> host_lut);

/// Print device pool memory usage.
void print_pool_stats(tile_buffers& bufs, cuda::device_ref device);

#endif // TILE_TRANSFER_H
