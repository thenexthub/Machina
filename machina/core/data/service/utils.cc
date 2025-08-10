/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#include "machina/core/data/service/utils.h"

#include <memory>
#include <string>

#include "machina/core/data/service/common.pb.h"
#include "machina/core/lib/io/record_reader.h"
#include "machina/core/lib/io/record_writer.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/path.h"

namespace machina {
namespace data {

absl::Status WriteDatasetDef(const std::string& path,
                             const DatasetDef& dataset_def) {
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(path, &file));
  io::RecordWriter writer(file.get());
  TF_RETURN_IF_ERROR(writer.WriteRecord(dataset_def.SerializeAsString()));
  return absl::OkStatus();
}

absl::Status ReadDatasetDef(const std::string& path, DatasetDef& dataset_def) {
  if (path.empty()) {
    return errors::InvalidArgument("Path is empty");
  }
  TF_RETURN_IF_ERROR(Env::Default()->FileExists(path));
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewRandomAccessFile(path, &file));
  io::RecordReader reader(file.get());
  uint64 offset = 0;
  tstring record;
  TF_RETURN_IF_ERROR(reader.ReadRecord(&offset, &record));
  if (!dataset_def.ParseFromString(record)) {
    return errors::DataLoss("Failed to parse dataset definition");
  }
  return absl::OkStatus();
}

}  // namespace data
}  // namespace machina
