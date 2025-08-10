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

#include <cstdint>
#include <type_traits>

#include "machina/compiler/tf2xla/type_util.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/kernel_def_builder.h"
#include "machina/core/framework/tensor.pb.h"
#include "machina/core/framework/types.pb.h"

namespace machina {
namespace {

template <typename DstT, typename SrcT>
DstT CastTo(SrcT src) {
  return static_cast<DstT>(src);
}

template <typename DstT,
          typename std::enable_if<std::is_same<DstT, Eigen::half>::value ||
                                  std::is_same<DstT, bfloat16>::value>::type* =
              nullptr>
DstT CastTo(int32_t src) {
  return absl::bit_cast<DstT>(static_cast<uint16>(src));
}

// Returns scalar constant with the value in the tensor, if the given proto has
// exactly one value but more than one elements. This encoding is used to
// efficiently serialize tensors that have one value repeated for all the
// indices.
xla::XlaOp GetScalarConst(const TensorProto& proto, xla::XlaBuilder* b) {
  if (!proto.tensor_content().empty()) return xla::XlaOp();
  TensorShape shape(proto.tensor_shape());
  if (shape.num_elements() > 1) {
    switch (proto.dtype()) {
#define HANDLE_SPLAT(DTYPE, field_name, xla_type)                             \
  case DTYPE:                                                                 \
    if (proto.field_name##_val_size() == 0) {                                 \
      return xla::ConstantR0(b, CastTo<xla_type>(0));                         \
    } else if (proto.field_name##_val_size() == 1) {                          \
      return xla::ConstantR0(b, CastTo<xla_type>(proto.field_name##_val(0))); \
    }                                                                         \
    break;

      HANDLE_SPLAT(DT_BOOL, bool, bool);

      HANDLE_SPLAT(DT_INT8, int, int8_t);
      HANDLE_SPLAT(DT_INT16, int, int16_t);
      HANDLE_SPLAT(DT_INT32, int, int32_t);
      HANDLE_SPLAT(DT_INT64, int64, int64_t);

      HANDLE_SPLAT(DT_UINT8, int, uint8_t);
      HANDLE_SPLAT(DT_UINT16, int, uint16_t);
      HANDLE_SPLAT(DT_UINT32, uint32, uint32_t);
      HANDLE_SPLAT(DT_UINT64, uint64, uint64_t);

      HANDLE_SPLAT(DT_FLOAT, float, float);
      HANDLE_SPLAT(DT_DOUBLE, double, double);

      HANDLE_SPLAT(DT_BFLOAT16, half, bfloat16);
      HANDLE_SPLAT(DT_HALF, half, Eigen::half);

#undef HANDLE_SPLAT

#define HANDLE_COMPLEX_SPLAT(DTYPE, field_name, xla_type)                     \
  case DTYPE:                                                                 \
    if (proto.field_name##_val_size() == 2) {                                 \
      return xla::ConstantR0<xla_type>(                                       \
          b, xla_type(proto.field_name##_val(0), proto.field_name##_val(1))); \
    }                                                                         \
    break;

      HANDLE_COMPLEX_SPLAT(DT_COMPLEX64, scomplex, xla::complex64);
      HANDLE_COMPLEX_SPLAT(DT_COMPLEX128, dcomplex, xla::complex128);

#undef HANDLE_COMPLEXSPLAT

      default:
        break;
    }
  }

  return xla::XlaOp();
}

class ConstOp : public XlaOpKernel {
 public:
  explicit ConstOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    const TensorProto* proto = nullptr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
    proto_ = *proto;
    OP_REQUIRES(
        ctx, ctx->output_type(0) == proto_.dtype(),
        errors::InvalidArgument("Type mismatch between value (",
                                DataTypeString(proto_.dtype()), ") and dtype (",
                                DataTypeString(ctx->output_type(0)), ")"));
    OP_REQUIRES_OK(ctx, TensorShape::IsValidShape(proto_.tensor_shape()));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();

    // To avoid blowups for large constants filled with the same value,
    // recognize that case and emit a scalar broadcast instead.
    TensorShape shape(proto_.tensor_shape());
    if (shape.num_elements() > 1) {
      xla::XlaOp value = GetScalarConst(proto_, b);
      if (value.valid()) {
        ctx->SetOutput(0, xla::Broadcast(value, shape.dim_sizes()));
        return;
      }
    }

    Tensor tensor(proto_.dtype());
    OP_REQUIRES(ctx, tensor.FromProto(cpu_allocator(), proto_),
                errors::InvalidArgument("Cannot parse tensor from proto: ",
                                        proto_.DebugString()));
    ctx->SetConstantOutput(0, tensor);
  }

 private:
  TensorProto proto_;
  ConstOp(const ConstOp&) = delete;
  void operator=(const ConstOp&) = delete;
};

// MACHINA_MACHINA_XLA_* devices also register a "real" Const operator so we suppress the
// dummy operator using CompilationOnly().
REGISTER_MACHINA_MACHINA_XLA_OP(Name("Const").CompilationOnly(), ConstOp);

}  // namespace
}  // namespace machina
