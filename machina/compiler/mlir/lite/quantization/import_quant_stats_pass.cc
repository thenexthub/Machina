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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"
#include "toolchain/ADT/APFloat.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringMap.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/Regex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "machina/compiler/mlir/lite/quantization/quantization_info.pb.h"
#include "machina/compiler/mlir/lite/quantization/quantization_passes.h"
#include "machina/compiler/mlir/machina/utils/import_utils.h"
#include "machina/compiler/mlir/machina/utils/location_utils.h"

// NOLINTNEXTLINE
static toolchain::cl::opt<std::string> quantize_stats(
    "quant-test-stats", toolchain::cl::value_desc("string"),
    toolchain::cl::desc("serialized quant info string. Only used in tests"),
    toolchain::cl::init(""));

//===----------------------------------------------------------------------===//
// The Pass to import quantization stats to the ops in a function. This requires
// a custom method to retrieve the unique name of the operation.

namespace mlir {
namespace quant {

using QuantParamsEntry = QuantizationInfo::QuantParams;

namespace {
class ImportQuantStatsPass
    : public PassWrapper<ImportQuantStatsPass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportQuantStatsPass)

  explicit ImportQuantStatsPass(OperationToName op_to_name)
      : op_to_name_(op_to_name) {}

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-import-stats";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Import quantization stats to the model";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<quant::QuantDialect,
                    quantfork::QuantizationForkDialect>();
  }

  // Parses the serialized quant stats protobuf and initialize the internal
  // data structure. This method must be called after the pass is created.
  bool ParseQuantStats(const std::string &stats_str);

 private:
  void ImportAsStatsOps(OpBuilder b, Operation *op, int index,
                        const QuantParamsEntry &info);

  void InsertStatsOpAtResult(OpBuilder b, Value res, ElementsAttr layer_stats,
                             ElementsAttr axis_stats, IntegerAttr axis);

  // If the index is out of range, this method returns false. Otherwise it
  // returns true if the value is a float tensor.
  bool IsQuantizableResult(Operation *op, int index) {
    if (index < 0 || index >= static_cast<int>(op->getNumResults()))
      return false;
    Value res = op->getResult(index);
    return isa<ShapedType>(res.getType()) &&
           isa<FloatType>(cast<ShapedType>(res.getType()).getElementType());
  }

  // A method to retrieve the name for the given op.
  OperationToName op_to_name_;

  // We split the normal names and regex names, since the former can use hash
  // map to lookup and the latter needs to iterate all the regex to find the
  // match.
  // The `int` in the following two containers are to specify the result index
  // of the given op. -1 indicates all the floating-point results.
  toolchain::StringMap<std::pair<int, const QuantParamsEntry>> name_to_info_;
  toolchain::StringMap<std::pair<int, const QuantParamsEntry>> regex_to_info_;
};
}  // namespace

bool ImportQuantStatsPass::ParseQuantStats(const std::string &stats_str) {
  QuantizationInfo quant_stats;
  if (!machina::LoadProtoFromBuffer(stats_str, &quant_stats).ok()) {
    return true;
  }

  for (const auto &entry : quant_stats.entries()) {
    if (!entry.name().empty()) {
      std::vector<std::string> name_and_port =
          absl::StrSplit(entry.name(), ':');
      int port = name_and_port.size() == 2 ? std::stoi(name_and_port[1]) : -1;
      name_to_info_.insert({name_and_port[0], {port, entry}});
    } else if (!entry.name_regex().empty()) {
      std::vector<std::string> name_and_port =
          absl::StrSplit(entry.name_regex(), ':');
      int port = name_and_port.size() == 2 ? std::stoi(name_and_port[1]) : -1;
      regex_to_info_.insert({name_and_port[0], {port, entry}});
    }
  }
  return false;
}

void ImportQuantStatsPass::InsertStatsOpAtResult(OpBuilder b, Value res,
                                                 ElementsAttr layer_stats,
                                                 ElementsAttr axis_stats,
                                                 IntegerAttr axis) {
  auto stats_op = b.create<quantfork::StatisticsOp>(
      b.getUnknownLoc(), res, layer_stats, axis_stats, axis);
  res.replaceAllUsesWith(stats_op);
  stats_op.getOperation()->replaceUsesOfWith(stats_op, res);
}

void ImportQuantStatsPass::ImportAsStatsOps(OpBuilder b, Operation *op,
                                            int index,
                                            const QuantParamsEntry &info) {
  if (info.params_size() == 0) return;

  SmallVector<APFloat, 4> min_maxs;
  min_maxs.reserve(info.params_size() * 2);
  for (const auto &param : info.params()) {
    toolchain::APFloat min(param.min_max().min());
    toolchain::APFloat max(param.min_max().max());
    min_maxs.push_back(min);
    min_maxs.push_back(max);
  }
  // The layer stats contain only the first min/max pairs.
  ElementsAttr layer_stats = DenseFPElementsAttr::get(
      RankedTensorType::get({2}, b.getF32Type()), {min_maxs[0], min_maxs[1]});
  ElementsAttr axis_stats;
  IntegerAttr axis;

  if (info.params_size() > 1) {
    SmallVector<int64_t, 4> axis_stats_shape{info.params_size(), 2};
    axis_stats = DenseFPElementsAttr::get(
        RankedTensorType::get(axis_stats_shape, b.getF32Type()), min_maxs);
    axis = b.getI64IntegerAttr(info.meta().quantize_axis());
  }

  b.setInsertionPointAfter(op);
  if (IsQuantizableResult(op, index)) {
    InsertStatsOpAtResult(b, op->getResult(index), layer_stats, axis_stats,
                          axis);
  } else {
    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      if (IsQuantizableResult(op, i)) {
        InsertStatsOpAtResult(b, op->getResult(i), layer_stats, axis_stats,
                              axis);
      }
    }
  }
}

void ImportQuantStatsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  OpBuilder builder(func);

  func.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>()) return;
    auto op_name = op_to_name_(op);

    // Check the named info collection first.
    auto it = name_to_info_.find(op_name);
    if (it != name_to_info_.end()) {
      ImportAsStatsOps(builder, op, it->second.first, it->second.second);
      return;
    }

    // Iterate all the regex names and matches the first one.
    for (auto &regex : regex_to_info_) {
      if (toolchain::Regex(regex.first()).match(op_name)) {
        ImportAsStatsOps(builder, op, regex.second.first, regex.second.second);
        break;
      }
    }
  });
}

// Creates an instance of the default quant parameters pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateImportQuantStatsPass(
    OperationToName op_to_name, const std::string &stats_str) {
  auto pass = std::make_unique<ImportQuantStatsPass>(op_to_name);
  if (pass->ParseQuantStats(stats_str)) return nullptr;
  return pass;
}

// Creates an instance pass to import quantization stats to the operations in
// the function. A custom method to get the name from the op is used because
// different dialect ops might have different ways to assign the name.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateImportQuantStatsPassForTFControlDialect(const std::string &stats_str) {
  auto get_name_func = [](Operation *op) {
    Location loc = machina::GetLocationWithoutOpType(op->getLoc());
    if (auto name = toolchain::dyn_cast<NameLoc>(loc)) {
      return name.getName().strref();
    } else if (auto fused_name = toolchain::dyn_cast<FusedLoc>(loc)) {
      for (auto sub_loc : fused_name.getLocations()) {
        if (auto named_sub_loc = toolchain::dyn_cast<NameLoc>(sub_loc)) {
          return named_sub_loc.getName().strref();
        }
      }
    }
    return toolchain::StringRef("");
  };

  return CreateImportQuantStatsPass(get_name_func, stats_str);
}

// Registers this pass with default values, only for test
static PassRegistration<ImportQuantStatsPass> pass([] {
  return CreateImportQuantStatsPassForTFControlDialect(quantize_stats);
});

}  // namespace quant
}  // namespace mlir
