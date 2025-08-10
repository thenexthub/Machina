/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/FormatVariadic.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-translate/Translation.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/lite/flatbuffer_export.h"
#include "machina/compiler/mlir/lite/flatbuffer_import.h"
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/compiler/mlir/machina/translate/tools/parsers.h"

using toolchain::cl::opt;

// Commandline flag to enable the control of flatbuffer import.
bool use_external_constant;

// Commandline flag to enable graph pruning.
bool experimental_prune_unreachable_nodes_unconditionally;

// NOLINTNEXTLINE
static opt<bool, true> use_external_constant_flag(
    "use-external-constant",
    toolchain::cl::desc("Use external constant during flatbuffer import"),
    toolchain::cl::location(use_external_constant), toolchain::cl::init(false));

// TODO(b/147111261): After the importer supports generic custom ops, we should
// change the flag to a more lightwise flag, e.g.
// "import_custom_ops_as_side_effect_free_ops", and let the MLIR DCE to prune
// the operations.
// NOLINTNEXTLINE
static opt<bool, true> experimental_prune_unreachable_nodes_unconditionally_flg(
    "experimental-prune-unreachable-nodes-unconditionally",
    toolchain::cl::desc("Prune nodes that are not ancestors of the output nodes."),
    toolchain::cl::location(experimental_prune_unreachable_nodes_unconditionally),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
static opt<std::string> input_arrays_flag(
    "input-arrays",
    toolchain::cl::desc(
        "List of input tensors, if different from the default inputs"),
    toolchain::cl::init(""));

// NOLINTNEXTLINE
static opt<std::string> output_arrays_flag(
    "output-arrays",
    toolchain::cl::desc(
        "List of output tensors, if different from the default outputs"),
    toolchain::cl::init(""));

using toolchain::cl::opt;

// These command line flags enable control of the translation implementation.
bool emit_builtin_tflite_ops;
bool emit_custom_ops;
bool emit_select_tf_ops;
bool lower_tensor_list_ops;
bool strip_debug_info;
bool use_buffer_offset;
bool emit_stablehlo_ops;
bool disable_vhlo_to_stablehlo;
bool serialize_debug_metadata;

// NOLINTNEXTLINE
static opt<bool, true> emit_builtin_tflite_ops_flag(
    "emit-builtin-tflite-ops",
    toolchain::cl::desc(
        "Emit TFLite built in operations in the generated TFLite model"),
    toolchain::cl::location(emit_builtin_tflite_ops), toolchain::cl::init(true));

// NOLINTNEXTLINE
static opt<bool, true> emit_select_tf_ops_flag(
    "emit-select-tf-ops",
    toolchain::cl::desc(
        "Emit Select TF operations (Flex ops) in the generated TFLite model"),
    toolchain::cl::location(emit_select_tf_ops), toolchain::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> emit_custom_ops_flag(
    "emit-custom-ops",
    toolchain::cl::desc("Emit Custom operations in the generated TFLite model"),
    toolchain::cl::location(emit_custom_ops), toolchain::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> lower_tensor_list_ops_flag(
    "lower-tensor-list-ops",
    toolchain::cl::desc("Lower the TensorList ops within the TFLite dialect"),
    toolchain::cl::location(lower_tensor_list_ops), toolchain::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> strip_debug_info_flag(
    "strip-debug-info", toolchain::cl::desc("Strip debug info during export"),
    toolchain::cl::location(strip_debug_info), toolchain::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> use_buffer_offset_flag(
    "use-buffer-offset",
    toolchain::cl::desc("store constant buffers outside of Flatbuffers"),
    toolchain::cl::location(use_buffer_offset), toolchain::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> emit_stablehlo_ops_flag(
    "emit-stablehlo-ops",
    toolchain::cl::desc("Whether serialize stablehlo ops or not"),
    toolchain::cl::location(emit_stablehlo_ops), toolchain::cl::init(false));

// Flatbuffer import by default will also perform vhlo to stablehlo legalization
// to hide serialization detail from user, but for debug purpose we need to be
// able to dump raw vhlo ops as well
// NOLINTNEXTLINE
static opt<bool, true> disable_vhlo_to_stablehlo_flag(
    "disable-vhlo-to-stablehlo",
    toolchain::cl::desc("Whether to deserialize to stablehlo ops or not"),
    toolchain::cl::location(disable_vhlo_to_stablehlo), toolchain::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> serialize_debug_metadata_flag(
    "serialize-debug-metadata",
    toolchain::cl::desc("Whether to serialize debug metadata or not"),
    toolchain::cl::location(serialize_debug_metadata), toolchain::cl::init(false));

// NOLINTNEXTLINE
static opt<bool> disable_buffer_deduping_flag(
    "disable-buffer-deduping",
    toolchain::cl::desc("Whether to disable buffer deduping or not"),
    toolchain::cl::init(false));

namespace mlir {
namespace {
static OwningOpRef<mlir::ModuleOp> FlatBufferFileToMlirTrans(
    toolchain::SourceMgr* source_mgr, MLIRContext* context,
    bool use_external_constant,
    bool experimental_prune_unreachable_nodes_unconditionally) {
  const toolchain::MemoryBuffer* input =
      source_mgr->getMemoryBuffer(source_mgr->getMainFileID());
  std::string error;
  auto loc =
      mlir::FileLineColLoc::get(context, input->getBufferIdentifier(), 0, 0);

  // Parses input/output names from command line options.
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  // Use output parser since we only have tensor names.
  if (!machina::ParseOutputArrayInfo(input_arrays_flag, &inputs).ok()) {
    return emitError(loc, "parsing input array info failed ")
               << input_arrays_flag,
           nullptr;
  }
  if (!machina::ParseOutputArrayInfo(output_arrays_flag, &outputs).ok()) {
    return emitError(loc, "parsing output array info failed ")
               << output_arrays_flag,
           nullptr;
  }
  return tflite::FlatBufferToMlir(
      absl::string_view(input->getBufferStart(), input->getBufferSize()),
      context, loc, use_external_constant, inputs, outputs,
      experimental_prune_unreachable_nodes_unconditionally,
      disable_vhlo_to_stablehlo);
}

static LogicalResult MlirToFlatBufferFileTranslateFunction(
    ModuleOp module, toolchain::raw_ostream& output) {
  std::string serialized_flatbuffer;
  std::unique_ptr<machina::OpOrArgNameMapper> op_or_arg_name_mapper;
  if (strip_debug_info) {
    op_or_arg_name_mapper =
        std::make_unique<machina::OpOrArgStripNameMapper>();
  } else {
    op_or_arg_name_mapper =
        std::make_unique<machina::OpOrArgLocNameMapper>();
  }
  tflite::FlatbufferExportOptions options;
  options.converter_flags.set_force_select_tf_ops(!emit_builtin_tflite_ops);
  options.converter_flags.set_enable_select_tf_ops(emit_select_tf_ops);
  options.converter_flags.set_allow_custom_ops(emit_custom_ops);
  options.converter_flags.set_use_buffer_offset(use_buffer_offset);
  options.op_or_arg_name_mapper = op_or_arg_name_mapper.get();
  options.converter_flags.set_serialize_debug_metadata(
      serialize_debug_metadata);
  options.disable_buffer_deduping = disable_buffer_deduping_flag.getValue();
  if (!tflite::MlirToFlatBufferTranslateFunction(
          module, options, &serialized_flatbuffer, emit_stablehlo_ops))
    return mlir::failure();

  output << serialized_flatbuffer;
  return success();
}
}  // namespace

static TranslateToMLIRRegistration FlatBufferFileToMlirTransReg(
    "tflite-flatbuffer-to-mlir", "tflite-flatbuffer-to-mlir",
    [](toolchain::SourceMgr& source_mgr, MLIRContext* context) {
      return FlatBufferFileToMlirTrans(
          &source_mgr, context, use_external_constant,
          experimental_prune_unreachable_nodes_unconditionally);
    });

static TranslateFromMLIRRegistration MLIRToFlatBufferTranslate(
    "mlir-to-tflite-flatbuffer", "mlir-to-tflite-flatbuffer",
    MlirToFlatBufferFileTranslateFunction, [](DialectRegistry& registry) {
      registry.insert<quant::QuantDialect,
                      quantfork::QuantizationForkDialect>();
      mlir::RegisterAllTensorFlowDialects(registry);
      registry.insert<TFL::TensorFlowLiteDialect>();
      registry.insert<arith::ArithDialect>();
      registry.insert<func::FuncDialect>();
      registry.insert<mlir::vhlo::VhloDialect>();
      registry.insert<mlir::stablehlo::StablehloDialect>();
    });
}  // namespace mlir
