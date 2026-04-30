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

/// @file
/// Implementation of supporting details: synthetic image generation and
/// printing/output helpers.  See detail.h for the interface.

#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/span>
#include <cuda/stream>

#include <cstdio>
#include <fstream>

#include "detail.h"

// ── Image generation ─────────────────────────────────────────────────

__device__ unsigned pixel_hash(int x, int y, unsigned seed)
{
  unsigned h = static_cast<unsigned>(x) * 1103515245u + static_cast<unsigned>(y) * 12345u + seed;
  h          = (h ^ (h >> 16)) * 0x45d9f3bu;
  h          = (h ^ (h >> 13)) * 0x85ebca6bu;
  return h ^ (h >> 16);
}

__device__ float hash01(int x, int y, unsigned seed)
{
  return static_cast<float>(pixel_hash(x, y, seed) & 0xFFFF) / 65535.0f;
}

__device__ float value_noise(float fx, float fy, float freq, unsigned seed)
{
  float sx = fx * freq, sy = fy * freq;
  int ix = static_cast<int>(floorf(sx)), iy = static_cast<int>(floorf(sy));
  float tx = sx - ix, ty = sy - iy;
  tx      = tx * tx * (3.0f - 2.0f * tx);
  ty      = ty * ty * (3.0f - 2.0f * ty);
  float a = hash01(ix, iy, seed) + (hash01(ix + 1, iy, seed) - hash01(ix, iy, seed)) * tx;
  float b = hash01(ix, iy + 1, seed) + (hash01(ix + 1, iy + 1, seed) - hash01(ix, iy + 1, seed)) * tx;
  return a + (b - a) * ty;
}

__device__ float fbm(float fx, float fy, int octaves, float freq, unsigned seed)
{
  float val = 0, amp = 1.0f, total = 0;
  for (int i = 0; i < octaves; ++i)
  {
    val += amp * value_noise(fx, fy, freq, seed + static_cast<unsigned>(i) * 7919u);
    total += amp;
    amp *= 0.5f;
    freq *= 2.0f;
  }
  return val / total;
}

struct generate_kernel
{
  template <typename Config>
  __device__ void operator()(Config config, cuda::std::span<pixel_t> out, int width, int row_offset)
  {
    auto tid = cuda::gpu_thread.rank(cuda::grid, config);
    if (tid >= out.size())
    {
      return;
    }

    int gx   = static_cast<int>(tid) % width;
    int gy   = static_cast<int>(tid) / width + row_offset;
    float fx = static_cast<float>(gx) / image_width;
    float fy = static_cast<float>(gy) / image_height;

    float val   = 4.0f + 3.0f * fy;
    float noise = (hash01(gx, gy, 0u) - 0.5f) * 6.0f;
    val += noise;

    float neb1_d = ((fx - 0.6f) * (fx - 0.6f) + (fy - 0.35f) * (fy - 0.35f)) / 0.06f;
    float neb1   = 40.0f * expf(-neb1_d);
    float neb2_d = ((fx - 0.35f) * (fx - 0.35f) + (fy - 0.6f) * (fy - 0.6f)) / 0.03f;
    float neb2   = 22.0f * expf(-neb2_d);

    float tex = fbm(fx, fy, 5, 8.0f, 42u);
    neb1 *= (0.5f + tex);
    neb2 *= (0.3f + 0.7f * tex);

    float dust      = fbm(fx + 0.1f, fy, 4, 6.0f, 137u);
    float dust_mask = fmaxf(0.0f, 1.0f - 2.0f * fabsf(dust - 0.5f));
    neb1 *= (1.0f - 0.6f * dust_mask * expf(-neb1_d * 2.0f));
    val += neb1 + neb2;

    constexpr int star_grid = 64;
    int cx                  = (gx / star_grid) * star_grid + star_grid / 2;
    int cy                  = (gy / star_grid) * star_grid + star_grid / 2;
    for (int dy = -1; dy <= 1; ++dy)
    {
      for (int dx = -1; dx <= 1; ++dx)
      {
        int scx     = cx + dx * star_grid;
        int scy     = cy + dy * star_grid;
        unsigned sh = pixel_hash(scx, scy, 9999u);
        if (static_cast<float>(sh & 0xFFFF) / 65535.0f < 0.08f)
        {
          float jx     = static_cast<float>((sh >> 4) & 0xFF) / 255.0f - 0.5f;
          float jy     = static_cast<float>((sh >> 12) & 0xFF) / 255.0f - 0.5f;
          float sx     = scx + jx * star_grid;
          float sy     = scy + jy * star_grid;
          float d2     = (gx - sx) * (gx - sx) + (gy - sy) * (gy - sy);
          float radius = 2.0f + static_cast<float>((sh >> 20) & 0xF);
          float bright = 80.0f + static_cast<float>((sh >> 24) & 0x7F);
          val += bright * expf(-d2 / (2.0f * radius * radius));
        }
      }
    }

    out[tid] = static_cast<pixel_t>(fminf(255.0f, fmaxf(0.0f, val)));
  }
};

// ── Image generation ─────────────────────────────────────────────────

void generate_image(cuda::stream_ref stream, tile_buffers& bufs, int num_tiles, cuda::std::span<pixel_t> host_preview)
{
  printf("=== Image generation (GPU) ===\n");
  cuda::timed_event gen_start{stream};

  for (int t = 0; t < num_tiles; ++t)
  {
    cuda::std::size_t offset = static_cast<cuda::std::size_t>(t) * bufs.tile_pixels;
    cuda::std::size_t count  = cuda::std::min(bufs.tile_pixels, image_pixels - offset);
    int tile_rows            = static_cast<int>(count / image_width);
    int row_offset           = t * static_cast<int>(bufs.tile_pixels / image_width);

    constexpr int block_size = 256;
    auto config              = cuda::distribute<block_size>(static_cast<int>(count));
    cuda::launch(stream, config, generate_kernel{}, bufs.dev_tile[0].first(count), image_width, row_offset);

    // Downscale the generated tile for the input preview while it's still on device.
    downscale_tile(stream, bufs, bufs.dev_tile[0].first(count), row_offset, tile_rows, host_preview);

    // Copy generated tile to host for the histogram pass.
    cuda::copy_bytes(stream, bufs.dev_tile[0].first(count), bufs.host_image.subspan(offset, count));
  }

  cuda::timed_event gen_end{stream};
  stream.sync();
  double gen_ms = (gen_end - gen_start).count() / 1e6;
  printf("  Generated %dx%d space observation (%.0f MB) in ~%.1f ms\n\n",
         image_width,
         image_height,
         image_pixels * sizeof(pixel_t) / (1024.0 * 1024.0),
         gen_ms);
}

// ── Printing / output helpers ────────────────────────────────────────

void print_device_info(cuda::device_ref dev, cuda::arch_traits_t traits, cuda::std::size_t total_mem)
{
  auto name = dev.name();
  auto cc   = dev.attribute(cuda::device_attributes::compute_capability);
  printf("\nSelected device %d: %.*s\n", dev.get(), static_cast<int>(name.size()), name.data());
  printf("  Compute capability: %d.%d\n", cc.major_cap(), cc.minor_cap());
  printf("  Total memory      : %.0f MB\n", total_mem / (1024.0 * 1024.0));
  printf("  Max threads/block : %d\n", traits.max_threads_per_block);
  printf("  Max shared memory : %zu bytes\n", traits.max_shared_memory_per_block);
}

void print_tile_plan(
  int tile_rows, int tile_alignment, int num_tiles, cuda::std::size_t budget, cuda::std::size_t total_mem)
{
  printf("\nTile plan:\n");
  printf("  Image          : %d x %d (%.0f MB)\n",
         image_width,
         image_height,
         image_pixels * sizeof(pixel_t) / (1024.0 * 1024.0));
  printf("  GPU budget     : %.0f MB (60%% of %.0f MB)\n", budget / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0));
  printf("  Tile rows      : %d (aligned to %d)\n", tile_rows, tile_alignment);
  printf("  Number of tiles: %d\n\n", num_tiles);
}

void print_allocation_info(
  cuda::std::size_t device_total, cuda::std::size_t gpu_budget, cuda::std::size_t tile_pixels, int tile_rows)
{
  printf("  Device pool: %.1f MB (initial), %.1f MB (max)\n",
         device_total / (1024.0 * 1024.0),
         gpu_budget / (1024.0 * 1024.0));
  printf("  Tile size: %zu pixels (%d rows x %d cols)\n", tile_pixels, tile_rows, image_width);
  printf("  Pinned host: image=%.1f MB, histogram=%zu bytes\n\n",
         image_pixels * sizeof(pixel_t) / (1024.0 * 1024.0),
         num_bins * sizeof(int));
}

void print_pool_stats(tile_buffers& bufs, cuda::device_ref device)
{
  auto reserved = bufs.device_pool.get().attribute(cuda::memory_pool_attributes::reserved_mem_current);
  auto used     = bufs.device_pool.get().attribute(cuda::memory_pool_attributes::used_mem_current);
  printf("  Device pool: reserved=%.1f MB, used=%.1f MB\n", reserved / (1024.0 * 1024.0), used / (1024.0 * 1024.0));
}

iqr_result compute_iqr(cuda::std::span<const int> hist, cuda::std::size_t total)
{
  auto find_percentile = [&](float pct) {
    cuda::std::size_t target     = static_cast<cuda::std::size_t>(total * pct);
    cuda::std::size_t cumulative = 0;
    for (cuda::std::size_t i = 0; i < hist.size(); ++i)
    {
      cumulative += hist[i];
      if (cumulative >= target)
      {
        return static_cast<int>(i);
      }
    }
    return static_cast<int>(hist.size()) - 1;
  };
  return {find_percentile(0.25f), find_percentile(0.75f)};
}

void print_pass_stats(double ms, long long total_selected, double mean_selected, float global_min, float global_max)
{
  // Note: times measured via cuda::timed_event are approximate GPU-side measurements.
  printf("  Pass time: ~%.1f ms\n", ms);
  printf("  Pixels above threshold: %lld / %zu (%.1f%%)\n",
         total_selected,
         image_pixels,
         100.0 * total_selected / image_pixels);
  printf("  Selected range: [%.4f, %.4f], mean=%.4f\n", global_min, global_max, mean_selected);
}

void print_sanity_check(iqr_result orig, iqr_result eq)
{
  bool ok = eq.width() > orig.width();
  printf("=== Sanity check ===\n");
  printf("  Original IQR (25th-75th):   [%d, %d] (span %d)\n", orig.p25, orig.p75, orig.width());
  printf("  Equalized IQR (25th-75th):  [%d, %d] (span %d)\n", eq.p25, eq.p75, eq.width());
  printf("  Equalization spread distribution: %s\n\n", ok ? "YES" : "NO");
}

void print_summary(int num_tiles, int tile_rows, double pass1_ms, double pass2_ms, bool ok)
{
  printf("=== Summary ===\n");
  printf("  Image:          %d x %d\n", image_width, image_height);
  printf("  Tiles:          %d (%d rows each)\n", num_tiles, tile_rows);
  printf("  Pass 1 (hist):  ~%.1f ms\n", pass1_ms);
  printf("  Pass 2 (stats): ~%.1f ms\n", pass2_ms);
  printf("  Total pipeline: ~%.1f ms\n", pass1_ms + pass2_ms);
  printf("  Result:         %s\n", ok ? "PASSED" : "FAILED");
}

void write_pgm(const char* filename, cuda::std::span<const pixel_t> data, int width, int height)
{
  std::ofstream file(filename, std::ios::binary);
  if (!file)
  {
    return;
  }
  file << "P5\n" << width << " " << height << "\n255\n";
  file.write(reinterpret_cast<const char*>(data.data()), data.size());
  printf("  Wrote %s (%d x %d)\n", filename, width, height);
}
