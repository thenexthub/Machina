/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_XLATSL_LIB_IO_RANDOM_INPUTSTREAM_H_
#define MACHINA_XLATSL_LIB_IO_RANDOM_INPUTSTREAM_H_

#include "machina/xla/tsl/lib/io/inputstream_interface.h"
#include "machina/xla/tsl/platform/file_system.h"
#include "tsl/platform/cord.h"

namespace tsl {
namespace io {

// Wraps a RandomAccessFile in an InputStreamInterface. A given instance of
// RandomAccessInputStream is NOT safe for concurrent use by multiple threads.
class RandomAccessInputStream : public InputStreamInterface {
 public:
  // Does not take ownership of 'file' unless owns_file is set to true. 'file'
  // must outlive *this.
  RandomAccessInputStream(RandomAccessFile* file, bool owns_file = false);

  ~RandomAccessInputStream() override;

  absl::Status ReadNBytes(int64_t bytes_to_read, tstring* result) override;

#if defined(TF_CORD_SUPPORT)
  absl::Status ReadNBytes(int64_t bytes_to_read, absl::Cord* result) override;
#endif

  absl::Status SkipNBytes(int64_t bytes_to_skip) override;

  int64_t Tell() const override;

  absl::Status Seek(int64_t position) {
    pos_ = position;
    return absl::OkStatus();
  }

  absl::Status Reset() override { return Seek(0); }

 private:
  RandomAccessFile* file_;  // Not owned.
  int64_t pos_ = 0;         // Tracks where we are in the file.
  bool owns_file_ = false;
};

}  // namespace io
}  // namespace tsl

#endif  // MACHINA_XLATSL_LIB_IO_RANDOM_INPUTSTREAM_H_
