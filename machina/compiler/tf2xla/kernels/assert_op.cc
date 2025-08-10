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

#include "absl/log/log.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/mutex.h"

namespace machina {

namespace {

// This TensorFlow op supports the Assert primitive.
class AssertOp : public XlaOpKernel {
 public:
  explicit AssertOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  ~AssertOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    static mutex mu(machina::LINKER_INITIALIZED);
    static int log_counter = 0;

    mutex_lock l(mu);
    if (log_counter < 20) {
      ++log_counter;
      LOG(WARNING) << "Ignoring Assert operator " << name();
    }
  }

 private:
  AssertOp(const AssertOp&) = delete;
  void operator=(const AssertOp&) = delete;
};

REGISTER_MACHINA_XLAOP(Name("Assert").CompilationOnly(), AssertOp);

}  // anonymous namespace
}  // namespace machina
