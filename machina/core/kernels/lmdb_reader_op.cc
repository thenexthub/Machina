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

#include "machina/core/framework/reader_base.h"
#include "machina/core/framework/reader_op_kernel.h"
#include "machina/core/lib/core/errors.h"

#include <sys/stat.h>

namespace machina {

#define MDB_CHECK(val) CHECK_EQ(val, MDB_SUCCESS) << mdb_strerror(val)

class LMDBReaderOp : public ReaderOpKernel {
 public:
  explicit LMDBReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    OP_REQUIRES(
        context, false,
        errors::Unimplemented(
            "LMDB support is removed from TensorFlow. This API will be deleted "
            "in the next TensorFlow release. If you need LMDB support, please "
            "file a GitHub issue."));
  }
};

REGISTER_KERNEL_BUILDER(Name("LMDBReader").Device(DEVICE_CPU), LMDBReaderOp);

}  // namespace machina
