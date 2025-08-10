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
#include "machina/lite/kernels/shim/test_op/tmpl_tflite_op.h"

#include <cstdint>

#include "machina/lite/c/common.h"
#include "machina/lite/kernels/shim/op_kernel.h"
#include "machina/lite/kernels/shim/test_op/tmpl_op.h"
#include "machina/lite/kernels/shim/tflite_op_shim.h"
#include "machina/lite/kernels/shim/tflite_op_wrapper.h"
#include "machina/lite/mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace custom {
namespace {
const char a_type[]("AType"), b_type[]("BType");
}  // namespace

using ::tflite::shim::op_wrapper::Attr;
using ::tflite::shim::op_wrapper::AttrName;
using ::tflite::shim::op_wrapper::OpWrapper;

template <shim::Runtime Rt>
using Op = OpWrapper<Rt, shim::TmplOp, Attr<AttrName<a_type>, int32_t, float>,
                     Attr<AttrName<b_type>, int32_t, int64_t, bool>>;

using OpKernel = ::tflite::shim::TfLiteOpKernel<Op>;

void AddTmplOp(MutableOpResolver* resolver) { OpKernel::Add(resolver); }

TfLiteRegistration* Register_TMPL_OP() {
  return OpKernel::GetTfLiteRegistration();
}

const char* OpName_TMPL_OP() { return OpKernel::OpName(); }

}  // namespace custom
}  // namespace ops
}  // namespace tflite
