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

// See docs in ../ops/io_ops.cc.

#include <memory>

#include "absl/strings/escaping.h"
#include "machina/core/framework/reader_base.h"
#include "machina/core/framework/reader_base.pb.h"
#include "machina/core/framework/reader_op_kernel.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/strings/str_util.h"
#include "machina/core/lib/strings/strcat.h"
#include "machina/core/platform/protobuf.h"

namespace machina {

class IdentityReader : public ReaderBase {
 public:
  explicit IdentityReader(const string& node_name)
      : ReaderBase(strings::StrCat("IdentityReader '", node_name, "'")) {}

  absl::Status ReadLocked(tstring* key, tstring* value, bool* produced,
                          bool* at_end) override {
    *key = current_work();
    *value = current_work();
    *produced = true;
    *at_end = true;
    return absl::OkStatus();
  }

  // Stores state in a ReaderBaseState proto, since IdentityReader has
  // no additional state beyond ReaderBase.
  absl::Status SerializeStateLocked(tstring* state) override {
    ReaderBaseState base_state;
    SaveBaseState(&base_state);
    SerializeToTString(base_state, state);
    return absl::OkStatus();
  }

  absl::Status RestoreStateLocked(const tstring& state) override {
    ReaderBaseState base_state;
    if (!ParseProtoUnlimited(&base_state, state)) {
      return errors::InvalidArgument("Could not parse state for ", name(), ": ",
                                     absl::CEscape(state));
    }
    TF_RETURN_IF_ERROR(RestoreBaseState(base_state));
    return absl::OkStatus();
  }
};

class IdentityReaderOp : public ReaderOpKernel {
 public:
  explicit IdentityReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    SetReaderFactory([this]() { return new IdentityReader(name()); });
  }
};

REGISTER_KERNEL_BUILDER(Name("IdentityReader").Device(DEVICE_CPU),
                        IdentityReaderOp);
REGISTER_KERNEL_BUILDER(Name("IdentityReaderV2").Device(DEVICE_CPU),
                        IdentityReaderOp);

}  // namespace machina
