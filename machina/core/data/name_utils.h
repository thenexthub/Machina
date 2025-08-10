/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#ifndef MACHINA_CORE_DATA_NAME_UTILS_H_
#define MACHINA_CORE_DATA_NAME_UTILS_H_

#include <vector>

#include "machina/core/lib/strings/strcat.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace data {
namespace name_utils {

extern const char kDelimiter[];
extern const char kDefaultDatasetDebugStringPrefix[];

struct OpNameParams {
  int op_version = 1;
};

struct DatasetDebugStringParams {
  template <typename... T>
  void set_args(T... input_args) {
    args = {static_cast<const strings::AlphaNum&>(input_args).data()...};
  }

  int op_version = 1;
  string dataset_prefix = "";
  std::vector<string> args;
};

struct IteratorPrefixParams {
  int op_version = 1;
  string dataset_prefix = "";
};

// Merge the given args in the format of "(arg1, arg2, ..., argn)".
//
// e.g. ArgsToString({"1", "2", "3"}) -> "(1, 2, 3)"; ArgsToString({}) -> "".
string ArgsToString(const std::vector<string>& args);

// Returns the dataset op name.
//
// e.g. OpName("Map") -> "MapDataset".
string OpName(const string& dataset_type);

// Returns the dataset op names.
//
// e.g. OpName(ConcatenateDatasetOp::kDatasetType, OpNameParams())
// -> "ConcatenateDataset"
//
// OpNameParams params;
// params.op_version = 2;
// OpName(ParallelInterleaveDatasetOp::kDatasetType, params)
// -> "ParallelInterleaveDatasetV2"
string OpName(const string& dataset_type, const OpNameParams& params);

// Returns a human-readable debug string for this dataset in the format of
// "FooDatasetOp(arg1, arg2, ...)::Dataset".
//
// e.g. DatasetDebugString("Map") -> "MapDatasetOp::Dataset";
string DatasetDebugString(const string& dataset_type);

// Returns a human-readable debug string for this dataset in the format of
// "FooDatasetOp(arg1, arg2, ...)::Dataset".
//
// e.g.
//  DatasetDebugStringParams range_params;
//  range_params.set_args(0, 10, 3);
//  DatasetDebugString(RangeDatasetOp::kDatasetType, range_params)
//  -> "RangeDatasetOp(0, 10, 3)::Dataset");
string DatasetDebugString(const string& dataset_type,
                          const DatasetDebugStringParams& params);

// Returns a string that identifies the sequence of iterators leading up to
// the iterator of this dataset.
//
// e.g. IteratorPrefix("Map", "Iterator::Range") -> "Iterator::Range::Map".
string IteratorPrefix(const string& dataset_type, const string& prefix);

// Returns a string that identifies the sequence of iterators leading up to
// the iterator of this dataset.
//
// e.g.
// IteratorPrefixParams params;
// params.op_version = 2;
// IteratorPrefix(BatchDatasetOp::KDatasetType, "Iterator::Range", params) ->
// "Iterator::Range::BatchV2".
string IteratorPrefix(const string& dataset_type, const string& prefix,
                      const IteratorPrefixParams& params);

}  // namespace name_utils
}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_DATA_NAME_UTILS_H_
