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
#include "machina/core/ir/importexport/load_proto.h"

#include "toolchain/Support/FileUtilities.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "toolchain/Support/raw_ostream.h"
#include "machina/core/ir/importexport/parse_text_proto.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/protobuf.h"

namespace machina {
namespace {
inline toolchain::StringRef StringViewToRef(absl::string_view view) {
  return {view.data(), view.size()};
}
}  // namespace

absl::Status LoadProtoFromBuffer(absl::string_view input,
                                 protobuf::Message* proto) {
  // Attempt to parse as text.
  if (mlir::tfg::ParseTextProto(input, "", proto).ok()) return absl::OkStatus();

  // Else attempt to parse as binary.
  return LoadProtoFromBuffer(input, static_cast<protobuf::MessageLite*>(proto));
}

absl::Status LoadProtoFromBuffer(absl::string_view input,
                                 protobuf::MessageLite* proto) {
  // Attempt to parse as binary.
  protobuf::io::ArrayInputStream binary_stream(input.data(), input.size());
  if (proto->ParseFromZeroCopyStream(&binary_stream)) return absl::OkStatus();

  LOG(ERROR) << "Error parsing Protobuf";
  return errors::InvalidArgument("Could not parse input proto");
}

template <class T>
absl::Status LoadProtoFromFileImpl(absl::string_view input_filename, T* proto) {
  const auto file_or_err =
      toolchain::MemoryBuffer::getFileOrSTDIN(StringViewToRef(input_filename));
  if (std::error_code error = file_or_err.getError()) {
    return errors::InvalidArgument(
        "Could not open input file ",
        string(input_filename.data(), input_filename.size()).c_str());
  }

  const auto& input_file = *file_or_err;
  absl::string_view content(input_file->getBufferStart(),
                            input_file->getBufferSize());
  return LoadProtoFromBuffer(content, proto);
}

absl::Status LoadProtoFromFile(absl::string_view input_filename,
                               protobuf::Message* proto) {
  return LoadProtoFromFileImpl(input_filename, proto);
}

absl::Status LoadProtoFromFile(absl::string_view input_filename,
                               protobuf::MessageLite* proto) {
  return LoadProtoFromFileImpl(input_filename, proto);
}

}  // namespace machina
