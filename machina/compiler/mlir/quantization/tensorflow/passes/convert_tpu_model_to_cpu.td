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
include "machina/compiler/mlir/machina/ir/tf_ops.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "machina/compiler/mlir/quantization/common/attrs_and_constraints.td"

// Combines the two variadic arguments ($in_tensors and $captured_tensors).
def GetBatchFunctionOpArgOperands:
    NativeCodeCall<"cast<TF::BatchFunctionOp>($0[0].getDefiningOp()).getArgOperands()">;

def CreateEmptyDictAttr : NativeCodeCall<"$_builder.getArrayAttr({})">;

// Replaces `TF_BatchFunctionOp` into `TF_PartitionedCallOp` that calls the
// same $f. This may be required, for example, when inlining is desired,
// because `TF_BatchFunctionOp` doesn't have the `CallOpInterface` trait.
def ReplaceBatchFunctionOpToPartitionedCallOp : Pat<
  (TF_BatchFunctionOp:$src_op_res
      $_, $_, $f, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_),
  (TF_PartitionedCallOp
      (GetBatchFunctionOpArgOperands $src_op_res),
      /*arg_attrs=*/(CreateEmptyDictAttr),
      /*res_attrs=*/(CreateEmptyDictAttr),
      $f,
      /*config=*/(CreateStringAttr<"">),
      /*config_proto=*/(CreateStringAttr<"">),
      /*executor_type=*/(CreateStringAttr<"">))>;
