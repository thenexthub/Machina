// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "machina_text/core/kernels/round_robin_trimmer_tflite.h"

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "machina/lite/c/common.h"
#include "machina/lite/kernels/shim/op_kernel.h"
#include "machina/lite/kernels/shim/tflite_op_shim.h"
#include "machina/lite/kernels/shim/tflite_op_wrapper.h"
#include "machina/lite/mutable_op_resolver.h"
#include "machina_text/core/kernels/round_robin_trimmer_kernel_template.h"

namespace tflite {
namespace ops {
namespace custom {
namespace text {
namespace {
const char splits_type[]("Tsplits"), vals_type[]("T");
}  // namespace

using ::tflite::shim::op_wrapper::Attr;
using ::tflite::shim::op_wrapper::AttrName;
using ::tflite::shim::op_wrapper::OpWrapper;

template <shim::Runtime Rt>
using GenerateMasksOp =
    OpWrapper<Rt, machina::text::RoundRobinGenerateMasksOp,
              Attr<AttrName<vals_type>, ::machina::tstring, float, double,
                   int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                   int64_t, uint64_t, bool>,
              Attr<AttrName<splits_type>, int32_t, int64_t>>;

extern "C" void AddRoundRobinGenerateMasks(
    tflite::MutableOpResolver* resolver) {
  tflite::shim::TfLiteOpKernel<GenerateMasksOp>::Add(resolver);
}

template <shim::Runtime Rt>
using TrimOp =
    OpWrapper<Rt, machina::text::RoundRobinTrimOp,
              Attr<AttrName<vals_type>, ::machina::tstring, float, double,
                   int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                   int64_t, uint64_t, bool>,
              Attr<AttrName<splits_type>, int32_t, int64_t>>;

extern "C" void AddRoundRobinTrim(tflite::MutableOpResolver* resolver) {
  tflite::shim::TfLiteOpKernel<TrimOp>::Add(resolver);
}

}  // namespace text
}  // namespace custom
}  // namespace ops
}  // namespace tflite
