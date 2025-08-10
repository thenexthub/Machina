/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_CORE_DATA_REWRITE_UTILS_H_
#define MACHINA_CORE_DATA_REWRITE_UTILS_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/core/platform/platform.h"

// On mobile we do not provide this functionality because not all of its
// dependencies are available there.
#if !defined(IS_MOBILE_PLATFORM)

#include <functional>
#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "machina/core/framework/dataset.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/grappler/grappler_item.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/platform/tstring.h"
#include "machina/core/platform/types.h"
#include "machina/core/protobuf/rewriter_config.pb.h"

namespace machina {
namespace data {

RewriterConfig CreateRewriterConfig(
    const absl::flat_hash_set<tstring>& optimizations,
    const absl::flat_hash_set<tstring>& optimizations_configs);

// Rewrites the input dataset using the given config. The rewritten_input
// stored in the core::RefCountPtr<DatasetBase>* output parameter is owned.
absl::Status RewriteDataset(OpKernelContext* ctx, const DatasetBase* input,
                            std::function<RewriterConfig(void)> config_factory,
                            bool record_fingerprint,
                            core::RefCountPtr<DatasetBase>* rewritten_input);

// Creates a grappler item for `graph_def`, which is required for graph
// optimization.
// `dataset_node` is the name of the node corresponding to the dataset.
// If `add_fake_sinks` is true, it adds fake sink node to graph and functions to
// allow rewriting the actual sink nodes.
// If `apply_optimizations` is true, general grappler optimizations at level
// `machina::OptimizerOptions::L1` are applied to the graph.
// TODO(b/118820916): When MetaOptimizer adds provisions for function retvals to
// be optimizable, we will no longer need to add fake nodes.
std::unique_ptr<machina::grappler::GrapplerItem> GetGrapplerItem(
    GraphDef* graph_def, std::string* dataset_node, bool add_fake_sinks,
    bool apply_optimizations = true);

// Returns the name of the node corresponding to the dataset. It is indicated by
// the symbolic `_Retval` node.
absl::StatusOr<std::string> GetDatasetNode(const GraphDef& graph_def);

// Like `GetDatasetNode` above, but returns the entire node object.
absl::StatusOr<NodeDef> GetDatasetNodeDef(const GraphDef& graph_def);

// Determines which optimizations should be applied.
//
// The result will contain any optimizations that are explicitly enabled, any
// default optimization that are not explicitly disabled, and any experiment
// that corresponds to an optimization as long as the optimization is not
// explicitly disabled.
absl::flat_hash_set<tstring> SelectOptimizations(
    const absl::flat_hash_set<string>& experiments,
    const absl::flat_hash_set<tstring>& optimizations_enabled,
    const absl::flat_hash_set<tstring>& optimizations_disabled,
    const absl::flat_hash_set<tstring>& optimizations_default);

}  // namespace data
}  // namespace machina
#endif  // !IS_MOBILE_PLATFORM

#endif  // MACHINA_CORE_DATA_REWRITE_UTILS_H_
