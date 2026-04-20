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
 * Tile transfer — buffer allocation and double-buffered host↔device copies.
 *
 * Demonstrates:
 *   cuda::device_memory_pool           custom device pool with planned capacity
 *   cuda::mr::shared_resource          wrap the pool for shared usage
 *   cuda::memory_pool_properties       initial_pool_size / release_threshold
 *   cuda::make_pinned_buffer           convenience for default pinned pool
 *   cuda::make_buffer                  typed, stream-ordered allocation
 *   cuda::no_init                      skip initialization
 *   cuda::copy_bytes                   host → device tile upload
 *   cuda::copy_configuration           source access order hints
 *   cuda::fill_bytes                   zero-clear device buffers
 *   buffer.subspan / buffer.first      span views without wrapping
 *   cuda::memory_pool_attributes       query reserved/used memory
 */

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>

#include <cstdio>

#include "tile_transfer.h"

tile_buffers
allocate_tile_buffers(cuda::stream_ref stream, cuda::device_ref device, int tile_rows, size_t gpu_budget, int num_tiles)
{
  printf("=== Tile buffer allocation ===\n");

  size_t tile_pixels = static_cast<size_t>(tile_rows) * image_width;

  // ── Custom device pool ─────────────────────────────────────────────
  // A single pool for both our buffers and CUB temporaries.  We pass
  // a copy of the shared_resource to CUB via the env, so all device
  // memory comes from the same place.  max_pool_size caps the total
  // device memory consumption to the budget computed from GPU capacity.
  size_t device_total =
    2 * tile_pixels * sizeof(pixel_t) // double-buffered pixel tiles
    + tile_pixels * sizeof(pixel_t) // equalized tile
    + tile_pixels * sizeof(float) // normalized float tile
    + tile_pixels * sizeof(float) // compacted output
    + sizeof(int) // num_selected
    + sizeof(float) // reduce output
    + num_bins * sizeof(pixel_t) // equalization LUT
    + num_bins * sizeof(int); // histogram

  cuda::memory_pool_properties props{};
  props.initial_pool_size = device_total;
  props.max_pool_size     = gpu_budget;

  // Wrap the device_memory_pool with shared_resource to allow for multiple buffers to share the same pool.
  auto device_pool = cuda::mr::shared_resource<cuda::device_memory_pool>(
    cuda::std::in_place_type<cuda::device_memory_pool>, device, props);

  printf("  Device pool: %.1f MB (initial), %.1f MB (max)\n",
         device_total / (1024.0 * 1024.0),
         gpu_budget / (1024.0 * 1024.0));

  // ── Allocate device buffers ────────────────────────────────────────
  auto dev_tile_0            = cuda::make_buffer<pixel_t>(stream, device_pool, tile_pixels, cuda::no_init);
  auto dev_tile_1            = cuda::make_buffer<pixel_t>(stream, device_pool, tile_pixels, cuda::no_init);
  auto dev_float             = cuda::make_buffer<float>(stream, device_pool, tile_pixels, cuda::no_init);
  auto dev_compact           = cuda::make_buffer<float>(stream, device_pool, tile_pixels, cuda::no_init);
  auto dev_nsel              = cuda::make_buffer<int>(stream, device_pool, 1, cuda::no_init);
  auto dev_hist              = cuda::make_buffer<int>(stream, device_pool, num_bins, cuda::no_init);
  auto dev_reduce_out        = cuda::make_buffer<float>(stream, device_pool, 3, cuda::no_init); // min, max, sum
  auto dev_lut               = cuda::make_buffer<pixel_t>(stream, device_pool, num_bins, cuda::no_init);
  auto dev_equalized         = cuda::make_buffer<pixel_t>(stream, device_pool, tile_pixels, cuda::no_init);
  size_t preview_tile_pixels = (tile_rows / preview_scale) * (image_width / preview_scale);
  auto dev_preview           = cuda::make_buffer<pixel_t>(stream, device_pool, preview_tile_pixels, cuda::no_init);

  printf("  Tile size: %zu pixels (%d rows x %d cols)\n", tile_pixels, tile_rows, image_width);

  // ── Pinned host buffers ────────────────────────────────────────────
  // make_pinned_buffer is equivalent to:
  //   cuda::make_buffer<T>(stream, cuda::pinned_default_memory_pool(), ...)
  // Default pools are owned by the runtime and only passed by reference. No need to wrap with shared_resource.
  auto host_image      = cuda::make_pinned_buffer<pixel_t>(stream, image_pixels, pixel_t{0});
  auto host_tile_hists = cuda::make_pinned_buffer<int>(stream, static_cast<size_t>(num_tiles) * num_bins, int{0});
  auto host_nsel       = cuda::make_pinned_buffer<int>(stream, 1, int{0});
  auto host_reduce     = cuda::make_pinned_buffer<float>(stream, 3, 0.0f); // min, max, sum

  printf("  Pinned host: image=%.1f MB, histogram=%zu bytes\n\n",
         image_pixels * sizeof(pixel_t) / (1024.0 * 1024.0),
         num_bins * sizeof(int));

  return {
    {cuda::std::move(dev_tile_0), cuda::std::move(dev_tile_1)},
    cuda::std::move(dev_float),
    cuda::std::move(dev_compact),
    cuda::std::move(dev_nsel),
    cuda::std::move(dev_hist),
    cuda::std::move(dev_reduce_out),
    cuda::std::move(dev_lut),
    cuda::std::move(dev_equalized),
    cuda::std::move(host_image),
    cuda::std::move(host_tile_hists),
    cuda::std::move(host_nsel),
    cuda::std::move(host_reduce),
    cuda::std::move(dev_preview),
    device_pool,
    tile_pixels,
  };
}

size_t upload_tile(cuda::stream_ref stream, tile_buffers& bufs, int slot, int tile_idx, int tile_rows)
{
  size_t offset = static_cast<size_t>(tile_idx) * bufs.tile_pixels;
  size_t count  = cuda::std::min(bufs.tile_pixels, image_pixels - offset);

  // Upload with a source access order hint: the pinned source data is
  // read in stream order (the default, shown here for demonstration).
  cuda::copy_configuration config{};
  config.src_access_order = cuda::source_access_order::stream;
  cuda::copy_bytes(stream, bufs.host_image.subspan(offset, count), bufs.dev_tile[slot].first(count), config);

  // Clear the per-tile histogram before accumulation.
  cuda::fill_bytes(stream, bufs.dev_histogram, ::cuda::std::uint8_t{0});

  return count;
}

void download_tile_histogram(cuda::stream_ref stream, tile_buffers& bufs, int tile_idx)
{
  // Copy device histogram into this tile's slot in the per-tile array.
  size_t offset = static_cast<size_t>(tile_idx) * num_bins;
  cuda::copy_bytes(stream, bufs.dev_histogram, bufs.host_tile_histograms.subspan(offset, num_bins));
}

void accumulate_histograms(tile_buffers& bufs, int num_tiles, cuda::std::span<int> result)
{
  for (int i = 0; i < num_bins; ++i)
  {
    result[i] = 0;
  }
  for (int t = 0; t < num_tiles; ++t)
  {
    size_t offset = static_cast<size_t>(t) * num_bins;
    for (int i = 0; i < num_bins; ++i)
    {
      result[i] += bufs.host_tile_histograms.get_unsynchronized(offset + i);
    }
  }
}

void upload_lut(cuda::stream_ref stream, tile_buffers& bufs, cuda::std::span<const pixel_t> host_lut)
{
  cuda::copy_bytes(stream, host_lut, bufs.dev_lut);
}

void print_pool_stats(tile_buffers& bufs, cuda::device_ref device)
{
  auto reserved = bufs.device_pool.get().attribute(cuda::memory_pool_attributes::reserved_mem_current);
  auto used     = bufs.device_pool.get().attribute(cuda::memory_pool_attributes::used_mem_current);
  printf("  Device pool: reserved=%.1f MB, used=%.1f MB\n", reserved / (1024.0 * 1024.0), used / (1024.0 * 1024.0));
}
