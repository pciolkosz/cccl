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
 * Image processing pipeline — main driver.
 *
 * Two-pass pipeline over a tiled grayscale space observation:
 *
 *   Pass 1 (histogram):
 *     For each tile, upload pixels to device, compute a per-tile histogram
 *     with CUB DeviceHistogram, download and accumulate into a global
 *     histogram.  Tiles are double-buffered across two streams.
 *
 *   Host interlude:
 *     Compute Otsu's threshold from the global histogram.  Build an
 *     equalization LUT.  Trim the memory pool.
 *
 *   Pass 2 (equalize + threshold + statistics + preview):
 *     For each tile, upload → equalize → normalize → histogram →
 *     compact → reduce → downscale preview.
 */

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/std/algorithm>
#include <cuda/std/span>
#include <cuda/stream>

#include <cstdio>

#include "device_setup.h"
#include "generate_image.h"
#include "image_pipeline.h"
#include "processing.h"
#include "tile_transfer.h"

// Write a PGM file from a pixel buffer.
static void write_pgm(const char* filename, const pixel_t* data, int width, int height)
{
  FILE* f = fopen(filename, "wb");
  if (!f)
  {
    return;
  }
  fprintf(f, "P5\n%d %d\n255\n", width, height);
  fwrite(data, 1, static_cast<size_t>(width) * height, f);
  fclose(f);
  printf("  Wrote %s (%d x %d)\n", filename, width, height);
}

int main()
{
  // ── 1. Device selection and tile sizing ────────────────────────────
  auto plan = select_device_and_plan();

  // ── 2. Allocate all buffers ────────────────────────────────────────
  cuda::stream stream{plan.device};
  auto bufs = allocate_tile_buffers(stream, plan.device, plan.tile_rows, plan.gpu_budget, plan.num_tiles);

  // ── 3. Generate a synthetic space observation on the GPU ───────────
  // In a real application, this would be replaced by loading actual
  // image data from disk or a sensor.  See generate_image.h.
  generate_image(stream, bufs, plan.num_tiles);

  // ── 4. Pass 1: histogram ───────────────────────────────────────────
  // Double-buffered across two streams:
  //
  //   stream_a: [upload tile 0] [histogram 0] [download 0]
  //   stream_b:                 [upload tile 1] [histogram 1] [download 1]
  //   stream_a:                                 [upload tile 2] [histogram 2] ...
  //
  // Each stream uses its own device tile buffer (slot 0 or 1).  An event
  // records when a slot's work finishes; the next use of that slot waits
  // on the event so we don't overwrite a tile that's still being
  // processed.  This lets upload of tile N+1 overlap with histogram
  // computation of tile N.
  printf("=== Pass 1: histogram ===\n");
  cuda::stream stream_a{plan.device};
  cuda::stream stream_b{plan.device};
  cuda::stream_ref streams[2] = {stream_a, stream_b};
  cuda::event slot_done[2]    = {cuda::event{plan.device}, cuda::event{plan.device}};

  cuda::timed_event pass1_start{stream_a};

  for (int t = 0; t < plan.num_tiles; ++t)
  {
    // Alternate between slot 0 (stream_a) and slot 1 (stream_b).
    int slot = t % 2;

    // Wait for this slot's previous work to finish before reusing its
    // device buffer.  On the first two iterations (t < 2) there is
    // nothing to wait for.
    if (t >= 2)
    {
      streams[slot].wait(slot_done[slot]);
    }

    size_t count = upload_tile(streams[slot], bufs, slot, t, plan.tile_rows);
    compute_histogram(streams[slot], bufs, slot, count);

    // Download into this tile's dedicated slot — no sync needed, each
    // tile writes to its own region of host_tile_histograms.
    download_tile_histogram(streams[slot], bufs, t);

    slot_done[slot].record(streams[slot]);
  }

  // Sync once after all tiles, then accumulate all histograms on the host.
  stream_a.sync();
  stream_b.sync();

  cuda::timed_event pass1_end{stream_a};
  stream_a.sync();
  double pass1_ms = (pass1_end - pass1_start).count() / 1e6;
  printf("  Histogram pass: %.1f ms\n", pass1_ms);

  // ── 5. Compute Otsu threshold and equalization LUT ─────────────────
  int original_hist[num_bins];
  auto global_hist_span = cuda::std::span<int>(original_hist, num_bins);
  accumulate_histograms(bufs, plan.num_tiles, global_hist_span);

  float otsu = compute_otsu_threshold(global_hist_span, image_pixels);
  printf("  Otsu threshold: %.4f (%d / 255)\n", otsu, static_cast<int>(otsu * 255));

  pixel_t host_lut[num_bins];
  build_equalization_lut(global_hist_span, image_pixels, cuda::std::span<pixel_t>(host_lut, num_bins));

  auto pinned_lut = cuda::make_pinned_buffer<pixel_t>(stream, num_bins, cuda::no_init);
  stream.sync();
  for (int i = 0; i < num_bins; ++i)
  {
    pinned_lut.get_unsynchronized(i) = host_lut[i];
  }
  upload_lut(stream, bufs, pinned_lut.subspan(0));
  printf("  Equalization LUT uploaded to device\n");

  // Zero the per-tile histograms before pass 2 reuses them for the
  // equalized image's histogram.
  cuda::fill_bytes(stream, bufs.host_tile_histograms, ::cuda::std::uint8_t{0});
  stream.sync();

  // ── 6. Pass 2: equalize + threshold + stats + preview ──────────────
  // Same double-buffered structure as pass 1.  Each tile goes through:
  //   upload → equalize → normalize → histogram → compact → reduce → downscale
  // The downscale writes a small preview tile to host memory so we
  // don't need to save the full equalized image.
  printf("=== Pass 2: equalize + threshold + statistics ===\n");
  cuda::timed_event pass2_start{stream_a};

  long long total_selected = 0;
  double global_sum        = 0;
  float global_min         = 1.0f;
  float global_max         = 0.0f;

  int pw               = image_width / preview_scale;
  int ph               = image_height / preview_scale;
  auto host_eq_preview = cuda::make_pinned_buffer<pixel_t>(stream, static_cast<size_t>(pw) * ph, pixel_t{0});
  stream.sync();

  for (int t = 0; t < plan.num_tiles; ++t)
  {
    int slot = t % 2;
    if (t >= 2)
    {
      streams[slot].wait(slot_done[slot]);
    }

    size_t count   = upload_tile(streams[slot], bufs, slot, t, plan.tile_rows);
    auto stats     = process_tile(streams[slot], bufs, slot, count, otsu);
    int tile_rows  = static_cast<int>(count / image_width);
    int row_offset = t * static_cast<int>(bufs.tile_pixels / image_width);

    downscale_tile(streams[slot], bufs, row_offset, tile_rows, host_eq_preview.subspan(0));
    download_tile_histogram(streams[slot], bufs, t);
    slot_done[slot].record(streams[slot]);

    total_selected += stats.num_selected;
    global_sum += stats.sum;
    if (stats.num_selected > 0)
    {
      global_min = cuda::std::min(global_min, stats.min_val);
      global_max = cuda::std::max(global_max, stats.max_val);
    }

    if ((t + 1) % 4 == 0 || t + 1 == plan.num_tiles)
    {
      printf("  Processed tile %d / %d\n", t + 1, plan.num_tiles);
    }
  }

  stream_a.sync();
  stream_b.sync();

  cuda::timed_event pass2_end{stream_a};
  stream_a.sync();
  double pass2_ms = (pass2_end - pass2_start).count() / 1e6;

  double mean_selected = (total_selected > 0) ? global_sum / total_selected : 0.0;

  printf("  Threshold pass: %.1f ms\n", pass2_ms);
  print_pool_stats(bufs, plan.device);
  printf("  Pixels above threshold: %lld / %zu (%.1f%%)\n",
         total_selected,
         image_pixels,
         100.0 * total_selected / image_pixels);
  printf("  Selected range: [%.4f, %.4f], mean=%.4f\n\n", global_min, global_max, mean_selected);

  // ── 7. Write downscaled preview images ─────────────────────────────
  printf("=== Output ===\n");

  // Input preview: downscale original pixels on GPU.
  auto host_input_preview = cuda::make_pinned_buffer<pixel_t>(stream, static_cast<size_t>(pw) * ph, pixel_t{0});
  stream.sync();

  for (int t = 0; t < plan.num_tiles; ++t)
  {
    size_t offset  = static_cast<size_t>(t) * bufs.tile_pixels;
    size_t count   = cuda::std::min(bufs.tile_pixels, image_pixels - offset);
    int tile_rows  = static_cast<int>(count / image_width);
    int row_offset = t * static_cast<int>(bufs.tile_pixels / image_width);

    // Upload and copy to dev_equalized so downscale_tile can read it.
    cuda::copy_bytes(stream, bufs.host_image.subspan(offset, count), bufs.dev_tile[0].first(count));
    cuda::copy_bytes(stream, bufs.dev_tile[0].first(count), bufs.dev_equalized.first(count));
    downscale_tile(stream, bufs, row_offset, tile_rows, host_input_preview.subspan(0));
  }
  stream.sync();
  write_pgm("input_preview.pgm", host_input_preview.data(), pw, ph);
  write_pgm("equalized_preview.pgm", host_eq_preview.data(), pw, ph);

  // ── 8. Sanity check ────────────────────────────────────────────────
  auto find_percentile = [](const int* hist, int bins, size_t total, float pct) {
    size_t target     = static_cast<size_t>(total * pct);
    size_t cumulative = 0;
    for (int i = 0; i < bins; ++i)
    {
      cumulative += hist[i];
      if (cumulative >= target)
      {
        return i;
      }
    }
    return bins - 1;
  };

  int orig_p25 = find_percentile(original_hist, num_bins, image_pixels, 0.25f);
  int orig_p75 = find_percentile(original_hist, num_bins, image_pixels, 0.75f);

  int equalized_hist[num_bins];
  accumulate_histograms(bufs, plan.num_tiles, cuda::std::span<int>(equalized_hist, num_bins));
  int eq_p25 = find_percentile(equalized_hist, num_bins, image_pixels, 0.25f);
  int eq_p75 = find_percentile(equalized_hist, num_bins, image_pixels, 0.75f);

  bool ok = (eq_p75 - eq_p25) > (orig_p75 - orig_p25);
  printf("=== Sanity check ===\n");
  printf("  Original IQR (25th–75th):   [%d, %d] (span %d)\n", orig_p25, orig_p75, orig_p75 - orig_p25);
  printf("  Equalized IQR (25th–75th):  [%d, %d] (span %d)\n", eq_p25, eq_p75, eq_p75 - eq_p25);
  printf("  Equalization spread distribution: %s\n\n", ok ? "YES" : "NO");

  // ── Summary ────────────────────────────────────────────────────────
  printf("=== Summary ===\n");
  printf("  Image:          %d x %d\n", image_width, image_height);
  printf("  Tiles:          %d (%d rows each)\n", plan.num_tiles, plan.tile_rows);
  printf("  Pass 1 (hist):  %.1f ms\n", pass1_ms);
  printf("  Pass 2 (stats): %.1f ms\n", pass2_ms);
  printf("  Total pipeline: %.1f ms\n", pass1_ms + pass2_ms);
  printf("  Result:         %s\n", ok ? "PASSED" : "FAILED");

  return ok ? 0 : 1;
}
