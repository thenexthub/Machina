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

#ifndef MACHINA_C_CHECKPOINT_READER_H_
#define MACHINA_C_CHECKPOINT_READER_H_

#include <memory>
#include <string>

#include "machina/c/tf_status_helper.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/types.h"
#include "machina/core/util/tensor_bundle/tensor_bundle.h"
#include "machina/core/util/tensor_slice_reader.h"

namespace machina {
namespace checkpoint {

class TensorSliceReader;

// A wrapper around BundleReader (for V2 checkpoints) and
// checkpoint::TensorSliceReader (for V1), that is more easily SWIG wrapped for
// other languages.
//
// The class currently only interacts with single-slice (i.e., non-partitioned)
// variables.
class CheckpointReader {
 public:
  CheckpointReader(const string& filename, TF_Status* status);

  bool HasTensor(const string& name) const;
  const string DebugString() const;

  // Returns a map from variable names to their shapes.  Slices of a partitioned
  // tensor are combined into a single entry.
  const TensorSliceReader::VarToShapeMap& GetVariableToShapeMap() const;

  // Returns a map from variable names to their data types.  Slices of a
  // partitioned tensor are combined into a single entry.
  const TensorSliceReader::VarToDataTypeMap& GetVariableToDataTypeMap() const;

  // Attempts to look up the tensor named "name" and stores the found result in
  // "out_tensor".
  void GetTensor(const string& name,
                 std::unique_ptr<machina::Tensor>* out_tensor,
                 TF_Status* out_status) const;

 private:
  // Uses "v2_reader_" to build "var name -> shape" and "var name -> data type"
  // maps; both owned by caller.
  // REQUIRES: "v2_reader_ != nullptr && v2_reader_.status().ok()".
  std::pair<std::unique_ptr<TensorSliceReader::VarToShapeMap>,
            std::unique_ptr<TensorSliceReader::VarToDataTypeMap> >
  BuildV2VarMaps();

  // Invariant: exactly one of "reader_" and "v2_reader_" is non-null.
  std::unique_ptr<TensorSliceReader> reader_;
  std::unique_ptr<BundleReader> v2_reader_;

  std::unique_ptr<TensorSliceReader::VarToShapeMap> var_to_shape_map_;
  std::unique_ptr<TensorSliceReader::VarToDataTypeMap> var_to_data_type_map_;

  CheckpointReader(const CheckpointReader&) = delete;
  void operator=(const CheckpointReader&) = delete;
};

}  // namespace checkpoint
}  // namespace machina

#endif  // MACHINA_C_CHECKPOINT_READER_H_
