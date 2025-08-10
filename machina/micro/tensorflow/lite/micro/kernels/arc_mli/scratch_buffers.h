/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

#ifndef MACHINA_LITE_MICRO_ARC_SCRATCH_BUFFERS_H_
#define MACHINA_LITE_MICRO_ARC_SCRATCH_BUFFERS_H_

#include "mli_api.h"  // NOLINT
#include "machina/lite/c/common.h"

namespace tflite {
namespace ops {
namespace micro {

void init_arc_scratch_buffers(void);
void* get_arc_scratch_buffer(int size);  // Function to assign fast memory
                                         // from one of 3 scratch buffers.

void get_arc_scratch_buffer_max_size(int* size);
void get_arc_scratch_buffer_two_max_sizes(int* size1, int* size2);

static inline bool inside_arc_dccm(void* p) {
#if core_config_dccm_present
  return ((unsigned)p >= core_config_dccm_base) &&
         ((unsigned)p < core_config_dccm_base + core_config_dccm_size);
#else
  return false;
#endif
}

static inline bool inside_arc_xccm(void* p) {
#if core_config_xy
  return ((unsigned)p >= core_config_xy_x_base) &&
         ((unsigned)p < core_config_xy_x_base + core_config_xy_size);
#else
  return false;
#endif
}

static inline bool inside_arc_yccm(void* p) {
#if core_config_xy_size
  return ((unsigned)p >= core_config_xy_y_base) &&
         ((unsigned)p < core_config_xy_y_base + core_config_xy_size);
#else
  return false;
#endif
}

static inline bool inside_arc_vccm(void* p) {
#if core_config_vec_mem_size
  return ((unsigned)p >= core_config_vec_mem_base) &&
         ((unsigned)p < core_config_vec_mem_base + core_config_vec_mem_size);
#else
  return false;
#endif
}

static inline bool inside_arc_ccm(void* p) {
  return inside_arc_dccm(p) || inside_arc_xccm(p) || inside_arc_yccm(p) ||
         inside_arc_vccm(p);
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_ARC_SCRATCH_BUFFERS_H_
