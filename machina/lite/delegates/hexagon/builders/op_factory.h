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
#ifndef MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_OP_FACTORY_H_
#define MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_OP_FACTORY_H_

#include "machina/lite/core/c/common.h"

namespace tflite {
namespace delegates {
namespace hexagon {
class GraphBuilder;
class OpBuilder;

OpBuilder* CreateArgMinMaxOpBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateActivationBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateArithmeticBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateMatMulWithConstWeightsOpBuilder(GraphBuilder* graph_builder,
                                                 int op_type);
OpBuilder* CreateConcatBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateConv2DBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateTransposeConv2DBuilder(GraphBuilder* graph_builder,
                                        int op_type);
OpBuilder* CreatePool2DBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateReshapeBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateSoftmaxBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateReduceBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateMirrorPadBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreatePadBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateResizeNearestNeighborBuilder(GraphBuilder* graph_builder,
                                              int op_type);
OpBuilder* CreateL2NormalizationBuilder(GraphBuilder* graph_builder,
                                        int op_type);
OpBuilder* CreateSplitBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateResizeBilinearOpBuilder(GraphBuilder* graph_builder,
                                         int op_type);
OpBuilder* CreateNegOpBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateTransposeBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateSpaceToDepthBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateBatchSeqBuilder(GraphBuilder* graph_builder, int op_type,
                                 int max_size_for_batch,
                                 TfLiteIntArray* input_batch_dimensions,
                                 TfLiteIntArray* output_batch_dimensions);
OpBuilder* CreateQuantizeBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateHardSwishBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateCastBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateMinMaxBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateSliceOpBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreatePackBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateMatMulOpBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateStridedSliceBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateSquaredDifferenceOpBuilder(GraphBuilder* graph_builder,
                                            int op_type);
OpBuilder* CreateRSqrtOpBuilder(GraphBuilder* graph_builder, int op_type);

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_OP_FACTORY_H_
