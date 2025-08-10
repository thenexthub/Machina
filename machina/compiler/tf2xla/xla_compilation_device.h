/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_MACHINA_MACHINA_XLA_COMPILATION_DEVICE_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_MACHINA_MACHINA_XLA_COMPILATION_DEVICE_H_

#include <memory>

#include "machina/core/common_runtime/local_device.h"
#include "machina/core/framework/device_base.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/mem.h"
#include "machina/core/public/session_options.h"

namespace machina {

// Class is defined in xla_compilation_device.cc, reference
// included here only so the XlaCompilationDevice allocator_ member can be
// declared.
class XlaCompilationAllocator;

// This is a 'dummy' TensorFlow device that is only used to execute a
// subgraph of XLA compilation Ops to construct a compiled version
// of the subgraph's computation. It has a 'dummy' allocator that
// backs each Tensor with an XlaExpression. The shape of the Tensor
// matches the shape of XlaExpression.
//
// We deliberately don't register a device factory because we *never*
// want placement to put Ops on a compilation device. The device is created
// manually, not using a factory.
//
// XLA compilation is not thread-safe. OpKernels registered on the
// XlaCompilationDevice must not use threads or concurrency.
class XlaCompilationDevice : public LocalDevice {
 public:
  XlaCompilationDevice(const SessionOptions& options, DeviceType type);

  ~XlaCompilationDevice() override;

  Allocator* GetAllocator(AllocatorAttributes attr) override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;

  absl::Status Sync() override;

  absl::Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                   const AllocatorAttributes alloc_attrs,
                                   Tensor* tensor) override;

 private:
  std::unique_ptr<XlaCompilationAllocator> allocator_;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_MACHINA_MACHINA_XLA_COMPILATION_DEVICE_H_
