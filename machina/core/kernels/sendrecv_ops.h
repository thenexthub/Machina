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

#ifndef MACHINA_CORE_KERNELS_SENDRECV_OPS_H_
#define MACHINA_CORE_KERNELS_SENDRECV_OPS_H_

#include "machina/core/framework/op_kernel.h"
#include "machina/core/platform/macros.h"

namespace machina {

class SendOp : public OpKernel {
 public:
  explicit SendOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

  string TraceString(const OpKernelContext& ctx, bool verbose) const override;

 private:
  string key_prefix_;
  Rendezvous::ParsedKey parsed_key_;
  bool hostmem_sendrecv_;

  SendOp(const SendOp&) = delete;
  void operator=(const SendOp&) = delete;
};

class RecvOp : public AsyncOpKernel {
 public:
  explicit RecvOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

  string TraceString(const OpKernelContext& ctx, bool verbose) const override;

 private:
  string key_prefix_;
  Rendezvous::ParsedKey parsed_key_;
  bool hostmem_sendrecv_;

  RecvOp(const RecvOp&) = delete;
  void operator=(const RecvOp&) = delete;
};

}  // end namespace machina

#endif  // MACHINA_CORE_KERNELS_SENDRECV_OPS_H_
