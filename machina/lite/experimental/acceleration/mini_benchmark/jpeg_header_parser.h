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
#ifndef MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_JPEG_HEADER_PARSER_H_
#define MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_JPEG_HEADER_PARSER_H_

#include <string>
#include <tuple>

#include "machina/lite/core/c/c_api_types.h"
#include "machina/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"
#include "machina/lite/experimental/acceleration/mini_benchmark/jpeg_common.h"
#include "machina/lite/string_type.h"
#include "machina/lite/string_util.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

// Extract the info in JpegHeader from the given buffer.
// Fails if the buffer doesn't contain a valid JPEG image in JFIF format.
Status ReadJpegHeader(const tflite::StringRef& jpeg_image_data,
                      JpegHeader* header);

// Writes into the given string the content of the JPEG image altered with
// the content of new_header.
// This is intented to be used in tests to forge existing images.
Status BuildImageWithNewHeader(const tflite::StringRef& orig_jpeg_image_data,
                               const JpegHeader& new_header,
                               std::string& new_image_data);

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite

#endif  // MACHINA_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_JPEG_HEADER_PARSER_H_
