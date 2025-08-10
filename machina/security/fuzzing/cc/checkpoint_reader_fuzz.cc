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

#include <map>
#include <memory>
#include <string>

#include "fuzztest/fuzztest.h"
#include "machina/c/checkpoint_reader.h"
#include "machina/c/tf_status.h"
#include "machina/c/tf_status_helper.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/core/framework/resource_handle.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_slice.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/framework/variant.h"
#include "machina/core/lib/gtl/cleanup.h"
#include "machina/core/lib/io/table_builder.h"
#include "machina/core/lib/io/table_options.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/file_system.h"
#include "machina/core/platform/resource_loader.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/tstring.h"
#include "machina/core/util/saved_tensor_slice.pb.h"
#include "machina/core/util/saved_tensor_slice_util.h"
#include "machina/security/fuzzing/cc/checkpoint_reader_fuzz_input.pb.h"

// This is a fuzzer for machina::checkpoint::CheckpointReader. LevelDB
// reading and proto parsing are already fuzz-tested, so there's no need to test
// them here.

namespace {

using ::machina::checkpoint::EncodeTensorNameSlice;
using ::machina::checkpoint::kSavedTensorSlicesKey;

void CreateCheckpoint(
    const std::string& filename,
    const machina::testing::CheckpointReaderFuzzInput& contents) {
  std::unique_ptr<machina::WritableFile> writable_file;
  TF_CHECK_OK(
      machina::Env::Default()->NewWritableFile(filename, &writable_file));
  machina::table::Options options;
  options.compression = machina::table::kNoCompression;
  machina::table::TableBuilder builder(options, writable_file.get());

  // Entries must be added in sorted order.
  {
    machina::SavedTensorSlices sts;
    *sts.mutable_meta() = contents.meta();
    builder.Add(kSavedTensorSlicesKey, sts.SerializeAsString());
  }
  std::map<std::string, const machina::SavedSlice*> entries;
  for (const machina::SavedSlice& saved_slice : contents.data()) {
    // The encoded tensor slice name is not included in the fuzz input since
    // it's difficult for the fuzzer to find the proper encoding, resulting in
    // lots of fruitless inputs with mismatched keys. Note that TensorSlice will
    // not currently crash with unverified data so long as it's only used by
    // EncodeTensorNameSlice.
    machina::TensorSlice slice(saved_slice.slice());
    entries.insert(
        {EncodeTensorNameSlice(saved_slice.name(), slice), &saved_slice});
  }
  machina::SavedTensorSlices sts;
  for (const auto& entry : entries) {
    *sts.mutable_data() = *entry.second;
    builder.Add(entry.first, sts.SerializeAsString());
  }
  TF_CHECK_OK(builder.Finish());
  TF_CHECK_OK(writable_file->Close());
}

int GetDataTypeSize(machina::DataType data_type) {
  // machina::DataTypeSize doesn't support several types.
  switch (data_type) {
    case machina::DT_STRING:
      return sizeof(machina::tstring);
    case machina::DT_VARIANT:
      return sizeof(machina::Variant);
    case machina::DT_RESOURCE:
      return sizeof(machina::ResourceHandle);
    default:
      return machina::DataTypeSize(data_type);
  }
}

static void FuzzTest(
    const machina::testing::CheckpointReaderFuzzInput& input) {
  // Using a ram file avoids disk I/O, speeding up the fuzzer.
  const std::string filename = "ram:///checkpoint";
  CreateCheckpoint(filename, input);
  // RamFileSystem::NewWritableFile doesn't remove existing files, so
  // expliciently ensure the checkpoint is deleted after each test.
  auto checkpoint_cleanup = machina::gtl::MakeCleanup([&filename] {
    TF_CHECK_OK(machina::Env::Default()->DeleteFile(filename));
  });

  machina::TF_StatusPtr status(TF_NewStatus());
  machina::checkpoint::CheckpointReader reader(filename, status.get());
  if (TF_GetCode(status.get()) != TF_OK) return;

  // Load each tensor in the input.
  std::unique_ptr<machina::Tensor> tensor;
  for (const auto& entry : input.meta().tensor()) {
    // Fuzz tests have a memory limit of 2 GB; skipping tensors over 1 GB is
    // sufficient to avoid OOMs.
    static constexpr double kMaxTensorSize = 1e9;
    auto data_type = reader.GetVariableToDataTypeMap().find(entry.name());
    auto shape = reader.GetVariableToShapeMap().find(entry.name());
    if (data_type != reader.GetVariableToDataTypeMap().end() &&
        shape != reader.GetVariableToShapeMap().end() &&
        static_cast<double>(GetDataTypeSize(data_type->second)) *
                shape->second.num_elements() <
            kMaxTensorSize) {
      reader.GetTensor(entry.name(), &tensor, status.get());
    }
  }
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
    .WithSeeds(fuzztest::ReadFilesFromDirectory<
      machina::testing::CheckpointReaderFuzzInput>(
        machina::GetDataDependencyFilepath(
          "machina/security/fuzzing/cc/checkpoint_reader_testdata")));

}  // namespace
