/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

// This file declares the functions and structures for memory I/O with libjpeg.
// These functions are not meant to be used directly, see jpeg_mem.h instead.
//
// Based on machina/core/lib/jpeg/jpeg_handle.h

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_IMAGE_JPEG_JPEG_HANDLE_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_IMAGE_JPEG_JPEG_HANDLE_H_

#include <cstdint>
#include <cstdio>

extern "C" {
#include "third_party/libjpeg_turbo/jerror.h"
#include "third_party/libjpeg_turbo/jpeglib.h"
}

namespace tfrt {
namespace image {
namespace jpeg {

// Handler for fatal JPEG library errors: clean up & return
void CatchError(j_common_ptr cinfo);

typedef struct {
  struct jpeg_destination_mgr pub;
  JOCTET *buffer;
  int bufsize;
  int datacount;
} MemDestMgr;

typedef struct {
  struct jpeg_source_mgr pub;
  const unsigned char *data;
  uint64_t datasize;
  bool try_recover_truncated_jpeg;
} MemSourceMgr;

void SetSrc(j_decompress_ptr cinfo, const void *data, uint64_t datasize,
            bool try_recover_truncated_jpeg);

}  // namespace jpeg
}  // namespace image
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_IMAGE_JPEG_JPEG_HANDLE_H_
