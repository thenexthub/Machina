/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_CORE_EXAMPLE_EXAMPLE_PARSER_CONFIGURATION_H_
#define MACHINA_CORE_EXAMPLE_EXAMPLE_PARSER_CONFIGURATION_H_

#include <string>
#include <vector>

#include "machina/core/example/example.pb.h"
#include "machina/core/example/example_parser_configuration.pb.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/types.h"
#include "machina/core/public/session.h"
#include "machina/core/util/example_proto_helper.h"
#include "machina/core/util/sparse/sparse_tensor.h"

// This is a set of helper methods that will make it possible to share
// machina::Example proto Tensor conversion code inside the ExampleParserOp
// OpKernel as well as in external code.
namespace machina {

// Given a graph and the node_name of a ParseExample op,
// extract the FixedLenFeature/VarLenFeature configurations.
absl::Status ExtractExampleParserConfiguration(
    const machina::GraphDef& graph, const string& node_name,
    machina::Session* session,
    std::vector<FixedLenFeature>* fixed_len_features,
    std::vector<VarLenFeature>* var_len_features);

// Given a config proto, ostensibly extracted via python,
// fill a vector of C++ structs suitable for calling
// the machina.Example -> Tensor conversion code.
absl::Status ExampleParserConfigurationProtoToFeatureVectors(
    const ExampleParserConfiguration& config_proto,
    std::vector<FixedLenFeature>* fixed_len_features,
    std::vector<VarLenFeature>* var_len_features);

}  // namespace machina

#endif  // MACHINA_CORE_EXAMPLE_EXAMPLE_PARSER_CONFIGURATION_H_
