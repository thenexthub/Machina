/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#include "machina_serving/util/status_util.h"

#include <string>

namespace machina {
namespace serving {

StatusProto ToStatusProto(const Status& status) {
  StatusProto status_proto;
  status_proto.set_error_code(
      static_cast<machina::error::Code>(status.code()));
  if (!status.ok()) {
    status_proto.set_error_message(std::string(status.message()));
  }
  return status_proto;
}

Status FromStatusProto(const StatusProto& status_proto) {
  return status_proto.error_code() == machina::error::OK
             ? Status()
             : Status(static_cast<absl::StatusCode>(status_proto.error_code()),
                      status_proto.error_message());
}

}  // namespace serving
}  // namespace machina
