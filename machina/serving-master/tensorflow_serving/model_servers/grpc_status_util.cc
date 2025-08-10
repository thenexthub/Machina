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

#include "machina_serving/model_servers/grpc_status_util.h"

#include "grpcpp/support/status_code_enum.h"
#include "absl/strings/str_join.h"

namespace machina {
namespace serving {

::grpc::Status ToGRPCStatus(const absl::Status& status) {
  const int kErrorMessageLimit = 1024;
  string error_message;
  if (status.message().length() > kErrorMessageLimit) {
    error_message = absl::StrCat(status.message().substr(0, kErrorMessageLimit),
                                 "...TRUNCATED");
  } else {
    error_message = status.message();
  }
  return ::grpc::Status(static_cast<grpc::StatusCode>(status.code()),
                        error_message);
}

}  // namespace serving
}  // namespace machina
