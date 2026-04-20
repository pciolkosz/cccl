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
 * Synthetic image generation — procedural space observation.
 *
 * This file exists purely for the example.  In a real application you
 * would replace generate_image() with code that loads actual data —
 * e.g., reading tiles from a FITS file (astronomy), a TIFF stack
 * (microscopy), or a satellite image archive.  The rest of the pipeline
 * (histogram, equalization, thresholding, statistics) is independent of
 * how the pixels are produced.
 *
 * The generated image is a deep-sky observation: mostly dark background
 * with sensor noise, two diffuse nebula lobes modulated by fractal
 * noise, dust lanes, and scattered gaussian-profile stars.
 *
 * Demonstrates:
 *   cuda::launch / cuda::distribute   kernel launch with hierarchy config
 *   cuda::copy_bytes                  device → pinned host transfer
 *   cuda::timed_event                 measure generation time
 *   buffer.first() / buffer.subspan() span views for partial copies
 */

#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda/stream>

#include <cstdio>

#include "generate_image.h"
#include "image_pipeline.h"

// ── Hash utilities for procedural generation ─────────────────────────
__device__ unsigned int pixel_hash(int x, int y, unsigned int seed)
{
  unsigned int h = static_cast<unsigned int>(x) * 1103515245u + static_cast<unsigned int>(y) * 12345u + seed;
  h              = (h ^ (h >> 16)) * 0x45d9f3bu;
  h              = (h ^ (h >> 13)) * 0x85ebca6bu;
  return h ^ (h >> 16);
}

__device__ float hash01(int x, int y, unsigned int seed)
{
  return static_cast<float>(pixel_hash(x, y, seed) & 0xFFFF) / 65535.0f;
}

__device__ float value_noise(float fx, float fy, float freq, unsigned int seed)
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

__device__ float fbm(float fx, float fy, int octaves, float freq, unsigned int seed)
{
  float val = 0, amp = 1.0f, total = 0;
  for (int i = 0; i < octaves; ++i)
  {
    val += amp * value_noise(fx, fy, freq, seed + static_cast<unsigned int>(i) * 7919u);
    total += amp;
    amp *= 0.5f;
    freq *= 2.0f;
  }
  return val / total;
}

// ── Generation kernel ────────────────────────────────────────────────
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

    // Background + sensor noise.
    float val   = 4.0f + 3.0f * fy;
    float noise = (hash01(gx, gy, 0u) - 0.5f) * 6.0f;
    val += noise;

    // Two nebula lobes with fractal texture and dust lanes.
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

    // Stars: gaussian blobs on a grid so they survive downscaling.
    constexpr int star_grid = 64;
    int cx                  = (gx / star_grid) * star_grid + star_grid / 2;
    int cy                  = (gy / star_grid) * star_grid + star_grid / 2;
    for (int dy = -1; dy <= 1; ++dy)
    {
      for (int dx = -1; dx <= 1; ++dx)
      {
        int scx         = cx + dx * star_grid;
        int scy         = cy + dy * star_grid;
        unsigned int sh = pixel_hash(scx, scy, 9999u);
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

// ── Public API ───────────────────────────────────────────────────────
void generate_image(cuda::stream_ref stream, tile_buffers& bufs, int num_tiles)
{
  printf("=== Image generation (GPU) ===\n");
  cuda::timed_event gen_start{stream};

  for (int t = 0; t < num_tiles; ++t)
  {
    size_t offset  = static_cast<size_t>(t) * bufs.tile_pixels;
    size_t count   = cuda::std::min(bufs.tile_pixels, image_pixels - offset);
    int tile_rows  = static_cast<int>(count / image_width);
    int row_offset = t * static_cast<int>(bufs.tile_pixels / image_width);

    constexpr int block_size = 256;
    auto config              = cuda::distribute<block_size>(static_cast<int>(count));
    cuda::launch(stream, config, generate_kernel{}, bufs.dev_tile[0].first(count), image_width, row_offset);

    cuda::copy_bytes(stream, bufs.dev_tile[0].first(count), bufs.host_image.subspan(offset, count));
  }

  cuda::timed_event gen_end{stream};
  stream.sync();
  double gen_ms = (gen_end - gen_start).count() / 1e6;
  printf("  Generated %dx%d space observation (%.0f MB) in %.1f ms\n\n",
         image_width,
         image_height,
         image_pixels * sizeof(pixel_t) / (1024.0 * 1024.0),
         gen_ms);
}
