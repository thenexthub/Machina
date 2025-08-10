/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_CORE_TFRT_SAVED_MODEL_UTILS_SERIALIZE_UTILS_H_
#define MACHINA_CORE_TFRT_SAVED_MODEL_UTILS_SERIALIZE_UTILS_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/utils/dump_mlir_util.h"
#include "machina/core/platform/status.h"
#include "machina/core/tfrt/mlrt/bytecode/executable.h"
#include "tsl/platform/env.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime

namespace machina {
namespace tfrt_stub {

// Serializes the BefBuffer into a file.
absl::Status SerializeBEF(const tfrt::BefBuffer &bef,
                          const std::string &filepath);

// Deserializes BEF file from filepath into a BEFBuffer.
absl::StatusOr<tfrt::BefBuffer> DeserializeBEFBuffer(
    const std::string &filepath);

// Serializes the MLRTBytecodeBuffer into a file.
absl::Status SerializeMLRTBytecode(const mlrt::bc::Buffer &byteCode,
                                   const std::string &filepath);

// Deserializes byte code from the given filepath into a MLRTBytecodeBuffer.
absl::StatusOr<mlrt::bc::Buffer> DeserializeMlrtBytecodeBuffer(
    const std::string &filepath);

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_SAVED_MODEL_UTILS_SERIALIZE_UTILS_H_
