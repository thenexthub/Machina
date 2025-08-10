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

#include <vector>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/InitLLVM.h"
#include "toolchain/Support/PrettyStackTrace.h"
#include "toolchain/Support/Regex.h"
#include "toolchain/TableGen/Main.h"
#include "toolchain/TableGen/Record.h"
#include "toolchain/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Operator.h"  // part of Codira Toolchain
#include "mlir/TableGen/Trait.h"  // part of Codira Toolchain

using toolchain::LessRecord;
using toolchain::raw_ostream;
using toolchain::Record;
using toolchain::RecordKeeper;
using mlir::tblgen::Operator;

// Helper macro that returns indented os.
#define OUT(X) os.indent((X))

// The function below has a non-constant reference as that is required by LLVM's
// TableGenMain.
// NOLINTNEXTLINE
static bool OpQuantSpecWriter(raw_ostream &os, const RecordKeeper &records) {
  toolchain::Regex acc_uniform_trait_regex{"AccumulatorUniformScale<([0-9]*),"};
  toolchain::Regex coeff_index_trait_regex{"AffineOpCoefficient<(-?[0-9]*),"};
  toolchain::Regex fixed_uniform_trait_regex{
      "FixedResultUniformScale<([0-9]+).*(true|false)>"};
  emitSourceFileHeader("Generated Ops Quant Spec Getters", os);

  // Retrieve all the definitions derived from Op definition and sort by record
  // name.
  std::vector<const Record *> defs = records.getAllDerivedDefinitions("Op");
  toolchain::sort(defs, LessRecord());

  OUT(0) << "static std::unique_ptr<OpQuantSpec> "
            "GetOpQuantSpec(mlir::Operation *op, bool "
            "disable_per_channel_for_dense_layers = false) {\n";
  // TODO(b/176258587): Move to OpTrait if this should be generalized.
  // Add special handling for LSTM.
  OUT(2) << "if (auto lstm_op = toolchain::dyn_cast<TFL::LSTMOp>(op)) {\n";
  OUT(4) << "return GetLstmOpQuantSpec<TFL::LSTMOp>(lstm_op);\n";
  OUT(2) << "} else if (auto lstm_op = "
            "toolchain::dyn_cast<TFL::UnidirectionalSequenceLSTMOp>(op)) {\n";
  OUT(4) << "return "
            "GetLstmOpQuantSpec<TFL::UnidirectionalSequenceLSTMOp>(lstm_op);\n";
  OUT(2) << "}\n";

  OUT(2) << "auto spec = std::make_unique<OpQuantSpec>();\n";
  toolchain::SmallVector<toolchain::StringRef, 3> matches;
  for (auto *def : defs) {
    Operator op(def);
    for (const auto t : op.getTraits()) {
      if (auto opTrait = toolchain::dyn_cast<mlir::tblgen::NativeTrait>(&t)) {
        auto trait_str = opTrait->getFullyQualifiedTraitName();
        if (!toolchain::StringRef{trait_str}.consume_front("::mlir::OpTrait::TFL::"))
          continue;

        OUT(2) << "if (auto tfl = toolchain::dyn_cast<" << op.getQualCppClassName()
               << ">(op)) {\n";
        // There is a "FixedResultUniformScale" trait, set the type for result.
        if (fixed_uniform_trait_regex.match(trait_str, &matches)) {
          OUT(4) << "for (int i = 0, e = op->getNumResults(); i != e; ++i)\n";
          OUT(6) << "spec->restricted_output_params[std::make_pair("
                 << matches[1] << ", " << matches[2]
                 << ")].push_back(tfl.::mlir::OpTrait::TFL::" << trait_str
                 << "<" << op.getQualCppClassName()
                 << ">::GetResultQuantizedType(i));\n";
          matches.clear();
        }
        // There is a "AccumulatorUniformScale" trait, set the type for bias.
        if (acc_uniform_trait_regex.match(trait_str, &matches)) {
          OUT(4) << "spec->biases_params.emplace(std::make_pair(" << matches[1]
                 << ", std::make_pair(tfl.GetAllNonBiasOperands(),"
                 << "GetUniformQuantizedTypeForBias)));\n";
          matches.clear();
        }
        // There is a "QuantChannelDim" trait, set the quantization dimension.
        if (coeff_index_trait_regex.match(trait_str, &matches)) {
          OUT(4) << "spec->coeff_op_quant_dim[tfl.GetCoefficientOperandIndex()"
                 << "] = toolchain::dyn_cast<TFL::FullyConnectedOp>(op) && "
                    "disable_per_channel_for_dense_layers ? -1 :  "
                    "tfl.GetQuantizationDim();\n";
          matches.clear();
        }

        OUT(2) << "}\n";
      }
    }
  }
  OUT(2) << "return spec;\n";
  OUT(0) << "}\n";
  return false;
}

int main(int argc, char **argv) {
  toolchain::InitLLVM y(argc, argv);
  toolchain::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &OpQuantSpecWriter);
}
