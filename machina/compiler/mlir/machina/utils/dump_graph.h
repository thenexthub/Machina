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
#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_DUMP_GRAPH_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_DUMP_GRAPH_H_

#include <optional>
#include <string>

#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "machina/core/framework/function.h"
#include "machina/core/graph/graph.h"
#include "machina/core/platform/status.h"

namespace machina {

struct MlirDumpConfig;

// Dumps 'graph_def' to a file, as textual IR. Returns the file name chosen.
//
// Note: This is for debugging use and is not optimized for performance.
absl::Status DumpTextualIRToFile(const MlirDumpConfig& config,
                                 const Graph& graph,
                                 const FunctionLibraryDefinition* flib_def,
                                 WritableFile* file);

// Config of the textual dump.
struct MlirDumpConfig {
  enum class Dialect {
    // Tensorflow Graph Dialect
    kTFG,
  };

  // The limit of element size that gets printed.
  MlirDumpConfig& elide_large_attributes(int large_element_limit = 16) {
    this->op_printing_flags.elideLargeElementsAttrs(large_element_limit);
    return *this;
  }

  // Enable printing of debug information. If 'pretty_form' is set to true,
  // debug information is printed in a more readable 'pretty' form but this
  // pretty form is not parsable (so only for human readability).
  MlirDumpConfig& emit_location_information(bool pretty_form = false) {
    this->op_printing_flags.enableDebugInfo(/*enable=*/true, pretty_form);
    return *this;
  }

  MlirDumpConfig& emit_dialect(Dialect dialect) {
    this->dialect = dialect;
    return *this;
  }

  // Op printing flags.
  mlir::OpPrintingFlags op_printing_flags = {};

  // The target MLIR dialect.
  Dialect dialect = Dialect::kTFG;
};

// Change DumpGraphToFile to dump MLIR textual IR instead of protobuf.
void UseMlirForGraphDump(const MlirDumpConfig& = {});

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_DUMP_GRAPH_H_
