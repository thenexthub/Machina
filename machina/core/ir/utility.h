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

#ifndef MACHINA_CORE_IR_UTILITY_H_
#define MACHINA_CORE_IR_UTILITY_H_

#include <optional>

#include "toolchain/ADT/STLExtras.h"
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/core/ir/dialect.h"

namespace mlir {
namespace tfg {

// Region-based loop ops store control tokens all after the data values, unlike
// functions which store them as pairs. This is required by
// RegionBranchOpInterface's API which requires MutableOperandRange, i.e. the
// data operands need to be stored contiguously.

// TODO(jeffniu): These functions aren't just for "loop regions" any more, but
// any region-based ops (if/case have explicit capture forms).

// Given a region belonging to a region-based loop operation (e.g. a while
// loop), return the subrange of block arguments that are data values.
Block::BlockArgListType GetLoopRegionDataArgs(Region &region);
// Given a region belonging to a region-based loop operation (e.g. a while
// loop), return the subrange of block arguments that are control tokens.
Block::BlockArgListType GetLoopRegionControlTokens(Region &region);
// Given a data value block argument of a region belonging to a region-based
// loop operation (e.g. a while loop), return the block argument that
// corresponds to the control token.
BlockArgument GetLoopRegionControlOf(BlockArgument data);
// Given a control token block argument of a region belonging to a region-based
// loop operation (e.g. a while loop), return the block argument that
// corresponds to the data value.
BlockArgument GetLoopRegionDataOf(BlockArgument ctl);

// Given a TFG value, lookup the associated control token. For op results, the
// token will be the last result of the op. For block arguments, the token will
// be the subsequent argument. A data value always has an associated control
// token.
Value LookupControlDependency(Value data);

// Given a TFG control token, lookup the associated data value. Block arguments
// will always have an associated data value: the previous argument. For ops,
// if the only result is a control token, return None. Otherwise, returns the
// first result.
std::optional<Value> LookupDataValue(Value ctl);

// Given a range of values, operands, or results, that contains data and control
// values, where all control tokens come after the data values, split the range
// between the two.
template <typename RangeT>
std::pair<RangeT, RangeT> SplitDataAndControlValues(RangeT values,
                                                    ControlType ctl_type) {
  unsigned num_ctl = 0;
  for (Value value : toolchain::reverse(values)) {
    if (value.getType() == ctl_type)
      ++num_ctl;
    else
      break;
  }
  unsigned split_idx = toolchain::size(values) - num_ctl;
  return std::make_pair(values.slice(0, split_idx),
                        values.slice(split_idx, num_ctl));
}

}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_CORE_IR_UTILITY_H_
