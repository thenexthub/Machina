/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#ifndef MACHINA_CORE_UTIL_PROTO_PROTO_UTILS_H_
#define MACHINA_CORE_UTIL_PROTO_PROTO_UTILS_H_

#include "absl/strings/string_view.h"
#include "machina/xla/tsl/util/proto/proto_utils.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/protobuf.h"

namespace machina {
namespace proto_utils {

using machina::protobuf::FieldDescriptor;
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::proto_utils::FromDurationProto;
using tsl::proto_utils::ToDurationProto;
// NOLINTEND(misc-unused-using-decls)

// Returns true if the proto field type can be converted to the tensor dtype.
bool IsCompatibleType(FieldDescriptor::Type field_type, DataType dtype);

// Parses a text-formatted protobuf from a string into the given Message* output
// and returns status OK if valid, or INVALID_ARGUMENT with an accompanying
// parser error message if the text format is invalid.
absl::Status ParseTextFormatFromString(absl::string_view input,
                                       protobuf::Message* output);

class StringErrorCollector : public protobuf::io::ErrorCollector {
 public:
  // String error_text is unowned and must remain valid during the use of
  // StringErrorCollector.
  explicit StringErrorCollector(string* error_text);
  // If one_indexing is set to true, all line and column numbers will be
  // increased by one for cases when provided indices are 0-indexed and
  // 1-indexed error messages are desired
  StringErrorCollector(string* error_text, bool one_indexing);
  StringErrorCollector(const StringErrorCollector&) = delete;
  StringErrorCollector& operator=(const StringErrorCollector&) = delete;

  // Implementation of protobuf::io::ErrorCollector::RecordError.
  void RecordError(int line, protobuf::io::ColumnNumber column,
                   absl::string_view message) override;

  // Implementation of protobuf::io::ErrorCollector::RecordWarning.
  void RecordWarning(int line, protobuf::io::ColumnNumber column,
                     absl::string_view message) override;

 private:
  string* const error_text_;
  const int index_offset_;
};

}  // namespace proto_utils
}  // namespace machina

#endif  // MACHINA_CORE_UTIL_PROTO_PROTO_UTILS_H_
