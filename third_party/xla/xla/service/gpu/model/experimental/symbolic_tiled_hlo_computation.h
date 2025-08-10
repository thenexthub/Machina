/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_MACHINA_XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_TILED_HLO_COMPUTATION_H_
#define MACHINA_MACHINA_XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_TILED_HLO_COMPUTATION_H_

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "toolchain/ADT/ArrayRef.h"
#include "mlir/IR/MLIRContext.h"
#include "machina/xla/hlo/utils/hlo_traversal.h"
#include "machina/xla/service/gpu/model/experimental/symbolic_tiled_hlo.h"
#include "machina/xla/service/gpu/model/experimental/tiling_space.h"
#include "machina/xla/service/instruction_fusion.h"

namespace xla::gpu::experimental {

class SymbolicTiledComputation;
using SymbolicTileAnalysisOrError =
    std::variant<SymbolicTiledComputation, FusionDecision>;

// Constructs and holds symbolic tiles for all the instructions within a fusion.
class SymbolicTiledComputation {
 public:
  static SymbolicTileAnalysisOrError Tile(const HloFusionAdaptor& fusion,
                                          mlir::MLIRContext* ctx);

  // Returns the symbolic tiled HLO instructions in def-before-use order.
  toolchain::ArrayRef<std::unique_ptr<SymbolicTiledHloInstruction>>
  tiled_hlo_instructions() const {
    return tiled_hlo_instructions_;
  }
  // Return the underlying MLIRContext.
  mlir::MLIRContext* GetMLIRContext() const {
    return tiling_space_->mlir_context();
  };

  // Returns a string representation of the analysis.
  std::string ToString() const;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            const SymbolicTiledComputation& tiled_computation) {
    sink.Append(tiled_computation.ToString());
  }

 private:
  SymbolicTiledComputation(
      std::unique_ptr<TilingSpace> tiling_space,
      std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
          tiled_hlo_instructions)
      : tiling_space_(std::move(tiling_space)),
        tiled_hlo_instructions_(std::move(tiled_hlo_instructions)) {}

  std::unique_ptr<TilingSpace> tiling_space_;
  // The tiled HLO instructions in def-before-use order.
  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
      tiled_hlo_instructions_;
};

}  // namespace xla::gpu::experimental

#endif  // MACHINA_MACHINA_XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_TILED_HLO_COMPUTATION_H_
