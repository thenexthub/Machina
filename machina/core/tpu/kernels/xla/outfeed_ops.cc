/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include <vector>

#include "machina/compiler/tf2xla/shape_util.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/shape.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/types.h"

namespace machina {

namespace {

// This TensorFlow op implements the XLA Outfeed primitive.
class OutfeedEnqueueOp : public XlaOpKernel {
 public:
  explicit OutfeedEnqueueOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::Shape xla_shape;
    OP_REQUIRES_OK(
        ctx, TensorShapeToXLAShape(dtype_, ctx->InputShape(0), &xla_shape));
    // Outfeed configuration is only needed for embedding outfeed.
    const string outfeed_config;
    xla::Outfeed(ctx->Input(0), xla_shape, outfeed_config);
  }

 private:
  DataType dtype_;

  OutfeedEnqueueOp(const OutfeedEnqueueOp&) = delete;
  void operator=(const OutfeedEnqueueOp&) = delete;
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("OutfeedEnqueue"), OutfeedEnqueueOp);

// This TensorFlow op implements the XLA Outfeed primitive for tuple types.
class OutfeedEnqueueTupleOp : public XlaOpKernel {
 public:
  explicit OutfeedEnqueueTupleOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> handles;
    std::vector<TensorShape> shapes;
    auto inputs = ctx->InputList("inputs", &handles, &shapes);

    std::vector<xla::Shape> xla_shapes;
    for (int i = 0; i < shapes.size(); ++i) {
      xla::Shape xla_shape;
      OP_REQUIRES_OK(ctx,
                     TensorShapeToXLAShape(dtypes_[i], shapes[i], &xla_shape));
      xla_shapes.push_back(xla_shape);
    }
    xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(xla_shapes);
    VLOG(1) << "OutfeedEnqueueTuple: "
            << xla::ShapeUtil::HumanStringWithLayout(tuple_shape);
    auto b = ctx->builder();
    auto tuple = xla::Tuple(b, handles);
    // Outfeed configuration is only needed for embedding outfeed.
    const string outfeed_config;
    xla::Outfeed(tuple, tuple_shape, outfeed_config);
  }

 private:
  DataTypeVector dtypes_;

  OutfeedEnqueueTupleOp(const OutfeedEnqueueTupleOp&) = delete;
  void operator=(const OutfeedEnqueueTupleOp&) = delete;
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("OutfeedEnqueueTuple"), OutfeedEnqueueTupleOp);

}  // anonymous namespace
}  // namespace machina
