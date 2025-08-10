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
#ifndef MACHINA_LITE_KERNELS_SHIM_TF_OP_SHIM_H_
#define MACHINA_LITE_KERNELS_SHIM_TF_OP_SHIM_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/framework/registration/registration.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/platform/status.h"
#include "machina/lite/kernels/shim/op_kernel.h"
#include "machina/lite/kernels/shim/shape.h"
#include "tsl/platform/macros.h"

// This file contains the TF adapter. That is, it takes a `OpKernelShim`
// class and provides a TF kernel out of it.

namespace tflite {
namespace shim {

// TF implementation of the methods during an op kernel initialization
class TfInitContext : public InitContext<TfInitContext> {
 public:
  explicit TfInitContext(const ::machina::OpKernelConstruction* context);
  // Read a given attribute
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const;

 private:
  const ::machina::OpKernelConstruction* context_;
};

// TF implementation of the methods during an op kernel invocation
class TfInvokeContext : public InvokeContext<TfInvokeContext> {
 public:
  explicit TfInvokeContext(::machina::OpKernelContext* context);
  // Read an input tensor
  ConstTensorViewOr GetInput(int idx) const;
  // Get a mutable output tensor
  TensorViewOr GetOutput(int idx, const Shape& shape) const;
  // Number of input tensors
  int NumInputs() const;
  // Number of output tensors
  int NumOutputs() const;

 private:
  ::machina::OpKernelContext* context_;
};

// TF implementation of the methods during shape inference
class TfShapeInferenceContext
    : public ShapeInferenceContext<TfShapeInferenceContext> {
 public:
  explicit TfShapeInferenceContext(
      ::machina::shape_inference::InferenceContext* context);
  // Read an input tensor shape
  ShapeOr GetInputShape(int idx) const;
  // Set an output tensor shape
  absl::Status SetOutputShape(int idx, const Shape& shape);
  // Read an input tensor during shape inference
  ConstTensorViewOr GetInputTensor(int idx) const;
  // Read a given attribute
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const;
  // Number of input tensors
  int NumInputs() const;
  // Number of output tensors
  int NumOutputs() const;

 private:
  ::machina::shape_inference::InferenceContext* context_;
};

// The adaptor between an op implementation (OpKernelShim subclass) and TF
// runtime
template <template <Runtime, typename...> typename Impl, typename... Ts>
class TfOpKernel : public ::machina::OpKernel {
 public:
  using ImplType = Impl<Runtime::kTf, Ts...>;

  explicit TfOpKernel(::machina::OpKernelConstruction* c)
      : OpKernel(c), impl_(std::make_unique<ImplType>()) {
    TfInitContext ctx(c);
    c->SetStatus(impl_->Init(&ctx));
  }

  // The main computation of the op
  void Compute(::machina::OpKernelContext* c) override {
    TfInvokeContext ctx(c);
    OP_REQUIRES_OK(c, impl_->Invoke(&ctx));
  }

  // Shape inference for the op.
  static absl::Status ShapeInference(
      ::machina::shape_inference::InferenceContext* c) {
    TfShapeInferenceContext ctx(c);
    return ImplType::ShapeInference(&ctx);
  }

  // The operation name
  static const char* OpName() { return ImplType::OpName(); }

 protected:
  std::unique_ptr<OpKernelShim<Impl, Runtime::kTf, Ts...>> impl_;
};

static_assert(::machina::shape_inference::InferenceContext::kUnknownDim ==
                  Shape::kUnknownDim,
              "The values must match.");
static_assert(::machina::shape_inference::InferenceContext::kUnknownRank ==
                  Shape::kUnknownRank,
              "The values must match.");

// Builds the OpDef to register the op with the TF runtime
template <typename Kernel>
::machina::register_op::OpDefBuilderWrapper CreateOpDefBuilderWrapper() {
  auto ret = ::machina::register_op::OpDefBuilderWrapper(
      Kernel::ImplType::OpName());
  for (const auto& input : Kernel::ImplType::Inputs()) ret = ret.Input(input);
  for (const auto& output : Kernel::ImplType::Outputs())
    ret = ret.Output(output);
  for (const auto& attr : Kernel::ImplType::Attrs()) ret = ret.Attr(attr);
  ret.SetShapeFn(Kernel::ShapeInference).Doc(Kernel::ImplType::kDoc);
  return ret;
}

template <>
struct ContextTypeForRuntime<Runtime::kTf> {
  using Init = TfInitContext;
  using Invoke = TfInvokeContext;
  using ShapeInference = TfShapeInferenceContext;
};

// Macros for defining an op. These are taken from op.h because they need to be
// slightly modified here.
#define REGISTER_OP_SHIM_IMPL(ctr, op_kernel_cls)                            \
  static ::machina::InitOnStartupMarker const register_op##ctr            \
      TF_ATTRIBUTE_UNUSED =                                                  \
          TF_INIT_ON_STARTUP_IF(SHOULD_REGISTER_OP(op_kernel_cls::OpName())) \
          << ::tflite::shim::CreateOpDefBuilderWrapper<op_kernel_cls>()

#define REGISTER_TF_OP_SHIM(...) \
  TF_ATTRIBUTE_ANNOTATE("tf:op") \
  TF_NEW_ID_FOR_INIT(REGISTER_OP_SHIM_IMPL, __VA_ARGS__)

}  // namespace shim
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_SHIM_TF_OP_SHIM_H_
