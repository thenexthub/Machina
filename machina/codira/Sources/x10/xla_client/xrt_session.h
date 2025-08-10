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

#ifndef X10_XLA_CLIENT_XRT_SESSION_H_
#define X10_XLA_CLIENT_XRT_SESSION_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "machina/compiler/xla/xla_client/debug_macros.h"
#include "machina/cc/client/client_session.h"
#include "machina/cc/framework/ops.h"
#include "machina/cc/framework/scope.h"
#include "machina/cc/ops/standard_ops.h"
#include "machina/compiler/xla/types.h"

namespace xla {

// Encapsulates an XRT session and its associated node cache. XrtSession are not
// thread safe, but are always accessed by one thread at a time. The
// XrtSessionCache will keep creating new sessions if not enough are available
// to satisfy the threads requests.
class XrtSession {
 public:
  // A cached node captures that single node, or the mini-graph root node,
  // together with the place-holders necessary to feed the node/sub-graph.
  // The end-point node can be either a machina Operation or an Output.
  struct CachedNode {
    CachedNode(machina::Output output,
               std::vector<machina::ops::Placeholder> holders)
        : holders(std::move(holders)) {
      outputs.push_back(std::move(output));
    }
    CachedNode(machina::Operation operation,
               std::vector<machina::ops::Placeholder> holders)
        : holders(std::move(holders)) {
      operations.push_back(std::move(operation));
    }
    CachedNode(std::vector<machina::Output> outputs,
               std::vector<machina::ops::Placeholder> holders)
        : outputs(std::move(outputs)), holders(std::move(holders)) {}
    CachedNode(std::vector<machina::Operation> operations,
               std::vector<machina::ops::Placeholder> holders)
        : operations(std::move(operations)), holders(std::move(holders)) {}

    std::vector<machina::Output> outputs;
    std::vector<machina::Operation> operations;
    std::vector<machina::ops::Placeholder> holders;
  };

  // The node cache holds a set of CachedNode of the same kind (by the means of
  // the NodeTypes entries).
  // The NodeCache access is not thread safe, but so is XrtSession.
  class NodeCache {
   public:
    bool Empty() const { return position_ >= nodes_.size(); }

    const CachedNode& Get() {
      XLA_CHECK_LT(position_, nodes_.size());
      ++position_;
      return *nodes_[position_ - 1];
    }

    void Add(std::shared_ptr<CachedNode> node) {
      nodes_.push_back(std::move(node));
    }

    void Rewind() { position_ = 0; }

   private:
    std::vector<std::shared_ptr<CachedNode>> nodes_;
    size_t position_ = 0;
  };

  explicit XrtSession(const machina::SessionOptions& session_options);

  const std::string& target() const { return target_; }

  machina::Scope* root() { return &root_; }

  machina::ClientSession* session() { return &session_; }

  NodeCache* GetNodeCache(const std::string& key) { return &node_cache_[key]; }

  void Reset();

  static std::string GetCacheKey(const std::string& op_name,
                                 const std::string& device);

 private:
  std::string target_;
  machina::Scope root_;
  machina::ClientSession session_;
  std::map<std::string, NodeCache> node_cache_;
};

}  // namespace xla

#endif  // X10_XLA_CLIENT_XRT_SESSION_H_
