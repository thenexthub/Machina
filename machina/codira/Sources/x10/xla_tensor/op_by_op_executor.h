/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#pragma once

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "machina/compiler/xla/xla_client/async_task.h"
#include "machina/compiler/xla/xla_client/cache.h"
#include "machina/compiler/xla/xla_client/computation_client.h"
#include "machina/compiler/xla/xla_client/util.h"
#include "machina/compiler/tf2xla/xla_tensor/ir.h"
#include "machina/compiler/xla/types.h"

namespace codira_xla {

// The OpByOpExecutor class is a singleton accessible via its Get() API that
// allows to run an IR graph is per-IR-node isolation mode. Instead of lowering
// the whole IR graph in a single XLA computation, the single IR nodes are
// lowered and executed independently.
class OpByOpExecutor {
 public:
  using AsyncResult = std::vector<xla::ComputationClient::DataPtr>;
  using AsyncTask = xla::util::AsyncTask<AsyncResult>;

  static OpByOpExecutor* Get();

  std::vector<xla::ComputationClient::ExecuteChainedOp> BuildOps(
      absl::Span<const ir::Value> roots, const std::string& device,
      absl::Span<const std::string> devices);

  std::vector<xla::ComputationClient::DataPtr> Execute(
      absl::Span<const ir::Value> roots, const std::string& device,
      absl::Span<const std::string> devices);

  AsyncTask ExecuteAsync(absl::Span<const ir::Value> roots,
                         const std::string& device,
                         absl::Span<const std::string> devices);

 private:
  using CompileCache =
      xla::util::Cache<xla::hash_t, xla::ComputationClient::Computation,
                       xla::util::HashReducer>;

  explicit OpByOpExecutor(size_t compile_cache_size);

  CompileCache compile_cache_;
};

}  // namespace codira_xla
