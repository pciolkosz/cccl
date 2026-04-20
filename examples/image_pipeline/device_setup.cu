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
 * Device setup — GPU selection and tile sizing.
 *
 * Demonstrates:
 *   cuda::devices                       iterate over all CUDA devices
 *   cuda::device_ref                    non-owning handle
 *   cuda::device_ref::name()            device name (returns string_view)
 *   cuda::device_ref::attribute()       per-device attribute queries
 *   cuda::device_attributes::*          strongly-typed compound attributes
 *   cuda::arch_traits_for()             per-architecture capabilities
 */

#include <cuda/devices>
#include <cuda/std/algorithm>

#include <cstdio>

#include <cuda_runtime.h>

#include "device_setup.h"
#include "image_pipeline.h"

device_plan select_device_and_plan()
{
  printf("=== Device selection ===\n");

  cuda::device_ref best = cuda::devices[0];
  size_t best_mem       = 0;

  for (auto dev : cuda::devices)
  {
    int sms = dev.attribute(cuda::device_attributes::multiprocessor_count);

    // TODO switch to total memory query once added
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev.get());
    size_t total_bytes = prop.totalGlobalMem;

    auto name = dev.name();
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

  auto cc        = best.attribute(cuda::device_attributes::compute_capability);
  auto traits    = cuda::arch_traits_for(cc);
  auto best_name = best.name();
  printf("\nSelected device %d: %.*s\n", best.get(), static_cast<int>(best_name.size()), best_name.data());
  printf("  Compute capability: %d.%d\n", cc.major_cap(), cc.minor_cap());
  printf("  Total memory      : %.0f MB\n", best_mem / (1024.0 * 1024.0));
  printf("  Max threads/block : %d\n", traits.max_threads_per_block);
  printf("  Max shared memory : %zu bytes\n", traits.max_shared_memory_per_block);

  // ── Compute tile rows from GPU memory budget ───────────────────────
  // We budget 60% of total GPU memory for the per-tile working set,
  // leaving room for the display driver and other allocations.
  size_t budget          = static_cast<size_t>(best_mem * 0.60);
  size_t overhead        = 128 * 1024 * 1024; // histogram, LUT, scalars, preview, CUB temporaries
  size_t bytes_per_pixel = 2 * sizeof(pixel_t) + sizeof(pixel_t) + 2 * sizeof(float);
  size_t max_tile_pixels = (budget - overhead) / bytes_per_pixel;
  int tile_rows          = static_cast<int>(max_tile_pixels / image_width);

  tile_rows = cuda::std::min(tile_rows, image_height);
  tile_rows = (tile_rows / 32) * 32;
  tile_rows = cuda::std::max(tile_rows, 32);

  int num_tiles = (image_height + tile_rows - 1) / tile_rows;

  printf("\nTile plan:\n");
  printf("  Image          : %d x %d (%.0f MB)\n",
         image_width,
         image_height,
         image_pixels * sizeof(pixel_t) / (1024.0 * 1024.0));
  printf("  GPU budget     : %.0f MB (60%% of %.0f MB)\n", budget / (1024.0 * 1024.0), best_mem / (1024.0 * 1024.0));
  printf("  Tile rows      : %d\n", tile_rows);
  printf("  Number of tiles: %d\n\n", num_tiles);

  return {best, tile_rows, num_tiles, budget};
}
