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

#ifndef DEVICE_SETUP_H
#define DEVICE_SETUP_H

#include <cuda/devices>

struct device_plan
{
  cuda::device_ref device;
  int tile_rows; // rows per tile
  int num_tiles; // total tiles
  size_t gpu_budget; // bytes available for the device pool
};

/// Select the GPU with the most memory, compute tile dimensions so that
/// the working set (two tiles for double-buffering + workspace) fits.
device_plan select_device_and_plan();

#endif // DEVICE_SETUP_H
