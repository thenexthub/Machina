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
#include "machina/lite/experimental/acceleration/mini_benchmark/jpeg_decompress_buffered_struct.h"

#include <gtest/gtest.h>

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {
namespace {

const int kSizeOfJpegDecompressStruct = sizeof(jpeg_decompress_struct);

TEST(JpegDecompressBufferedStructTest,
     ExpectInitializationSizeMatchesStructSize) {
  JpegDecompressBufferedStruct buffered_struct(kSizeOfJpegDecompressStruct);
  EXPECT_EQ(buffered_struct.size(), kSizeOfJpegDecompressStruct);
}

TEST(JpegDecompressBufferedStructTest,
     StructWithSizeGreaterThanCompiledStruct) {
  int excess_bytes = 16;
  JpegDecompressBufferedStruct buffered_struct(kSizeOfJpegDecompressStruct +
                                               excess_bytes);
  EXPECT_EQ(buffered_struct.size(), kSizeOfJpegDecompressStruct + excess_bytes);
  const char* buffer = buffered_struct.buffer();
  ASSERT_NE(buffer, nullptr);
  while (excess_bytes--) {
    EXPECT_EQ(
        (unsigned char)(buffer[kSizeOfJpegDecompressStruct + excess_bytes]),
        '\0');
  }
}

TEST(JpegDecompressBufferedStructTest, StructWithSizeLessThanCompiledStruct) {
  JpegDecompressBufferedStruct buffered_struct(kSizeOfJpegDecompressStruct -
                                               16);
  EXPECT_EQ(buffered_struct.size(), kSizeOfJpegDecompressStruct);
}

}  // namespace
}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
