/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_XLAHLO_TRANSLATE_STABLEHLO_TO_HLO_TRANSLATE_H_
#define MACHINA_XLAHLO_TRANSLATE_STABLEHLO_TO_HLO_TRANSLATE_H_

#include <memory>
#include <utility>

#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/raw_os_ostream.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace xla {

mlir::LogicalResult StablehloToHloTranslateFunction(mlir::ModuleOp module,
                                                    toolchain::raw_ostream& output,
                                                    bool emit_return_tuple,
                                                    bool emit_use_tuple_arg);

mlir::LogicalResult StablehloToHloTextTranslateFunction(
    mlir::ModuleOp module, toolchain::raw_ostream& output, bool emit_return_tuple,
    bool emit_use_tuple_arg, bool print_layouts, bool print_large_constants,
    bool print_sugar, bool via_builder, bool with_layouts);

// Translate the StableHLO program in in-memory file 'buffer' to a HLO program
// written in a file represented with handle 'output_stream';
mlir::LogicalResult StablehloToHloTextMain(
    std::unique_ptr<toolchain::MemoryBuffer> buffer,
    toolchain::raw_ostream& output_stream, bool emit_return_tuple,
    bool emit_use_tuple_arg, bool print_layouts, bool print_large_constants,
    bool print_sugar, bool via_builder, bool with_layouts);

}  // namespace xla

#endif  // MACHINA_XLAHLO_TRANSLATE_STABLEHLO_TO_HLO_TRANSLATE_H_
