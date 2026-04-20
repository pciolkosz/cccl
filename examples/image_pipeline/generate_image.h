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

#ifndef GENERATE_IMAGE_H
#define GENERATE_IMAGE_H

#include <cuda/stream>

#include "tile_transfer.h"

/// Generate a synthetic test image on the GPU, tile by tile.
///
/// In a real application this function would be replaced by loading
/// actual image data — e.g., reading tiles from a FITS file (astronomy),
/// a TIFF stack (microscopy), or a satellite image archive.  The rest
/// of the pipeline (histogram, equalization, thresholding, statistics)
/// would remain unchanged.
///
/// For this example we generate a procedural space observation: dark
/// background with sensor noise, two diffuse nebula lobes modulated by
/// fractal noise, dust lanes, and scattered gaussian-profile stars.
void generate_image(cuda::stream_ref stream, tile_buffers& bufs, int num_tiles);

#endif // GENERATE_IMAGE_H
