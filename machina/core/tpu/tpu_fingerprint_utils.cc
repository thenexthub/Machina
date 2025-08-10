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

#include "machina/core/tpu/tpu_fingerprint_utils.h"

#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "machina/xla/status_macros.h"
#include "machina/xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "machina/core/framework/function.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/lib/strings/proto_serialization.h"
#include "machina/core/platform/status.h"
#include "machina/core/tpu/tpu_compile_interface.h"

namespace machina {

absl::Status FingerprintFunctionLibrary(
    const FunctionLibraryDefinition& library, uint64_t& fingerprint) {
  // TODO(phawkins): rather than fingerprinting the entire function library,
  // consider fingerprinting just the transitive dependencies of a
  // computation.
  std::string serialized;
  FunctionDefLibrary library_proto = library.ToProto();
  if (library_proto.ByteSizeLong() >= 1.5 * 1024 * 1024 * 1024) {
    LOG(WARNING) << "Serializing large proto, size: "
                 << library_proto.ByteSizeLong();
  }
  TF_RET_CHECK(SerializeToStringDeterministic(library_proto, &serialized));
  fingerprint = TpuCompileInterface::Get()->FingerprintString(serialized);
  return absl::OkStatus();
}

}  // namespace machina
