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

#include "machina/compiler/tf2xla/xla_tensor/debug_util.h"

#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>

#include "absl/container/node_hash_set.h"
#include "absl/strings/str_split.h"
#include "machina/compiler/tf2xla/xla_tensor/ir.h"
#include "machina/compiler/tf2xla/xla_tensor/ir_dump_util.h"
#include "machina/compiler/tf2xla/xla_tensor/ir_util.h"
#include "machina/compiler/tf2xla/xla_tensor/codira_backtrace.h"
#include "machina/compiler/xla/xla_client/debug_macros.h"
#include "machina/compiler/xla/xla_client/device.h"
#include "machina/compiler/xla/xla_client/sys_util.h"
#include "machina/compiler/xla/xla_client/unique.h"

namespace codira_xla {
namespace {

DebugUtil::GraphFormat DefaultGraphFormat() {
  std::string fmt_str =
      xla::sys_util::GetEnvString("XLA_SAVE_TENSORS_FMT", "text");
  if (fmt_str == "text") {
    return DebugUtil::GraphFormat::kText;
  } else if (fmt_str == "hlo") {
    return DebugUtil::GraphFormat::kHlo;
  } else if (fmt_str == "dot") {
    return DebugUtil::GraphFormat::kDot;
  }
  XLA_ERROR() << "Invalid save graph format: " << fmt_str;
}

absl::node_hash_set<std::string>* LoadExperiments() {
  std::unique_ptr<absl::node_hash_set<std::string>> xset =
      absl::make_unique<absl::node_hash_set<std::string>>();
  std::string experiments = xla::sys_util::GetEnvString("XLA_EXPERIMENTAL", "");
  std::vector<std::string> experiment_list = absl::StrSplit(experiments, ':');
  for (auto& name : experiment_list) {
    xset->insert(name);
  }
  return xset.release();
}

}  // namespace

DebugUtil::GraphFormat DebugUtil::GetDefaultGraphFormat() {
  static GraphFormat format = DefaultGraphFormat();
  return format;
}

std::string DebugUtil::GetTensorsGraphInfo(absl::Span<const XLATensor> tensors,
                                           const std::vector<size_t>* indices,
                                           GraphFormat format) {
  std::vector<const ir::Node*> root_nodes;
  std::vector<ir::Value> root_values;
  std::vector<xla::hash_t> root_hashes;
  xla::util::Unique<Device> unique_device;
  if (indices != nullptr) {
    for (auto index : *indices) {
      const XLATensor& tensor = tensors[index];
      ir::Value ir_value = tensor.CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_hashes.push_back(ir_value.hash());
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor.GetDevice());
      }
    }
  } else {
    for (auto& tensor : tensors) {
      ir::Value ir_value = tensor.CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_hashes.push_back(ir_value.hash());
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor.GetDevice());
      }
    }
  }
  std::stringstream ss;
  ss << "TensorsGraphInfo:\n";
  ss << GetSwiftFrames();
  ss << "\nHashes: (";
  for (size_t i = 0; i < root_hashes.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << xla::util::HexHash(root_hashes[i]);
  }
  ss << ")\n";
  std::string graph_str;
  if (format == GraphFormat::kText) {
    graph_str = ir::DumpUtil::ToText(root_nodes);
  } else if (format == GraphFormat::kDot) {
    graph_str = ir::DumpUtil::ToDot(root_nodes);
  } else if (format == GraphFormat::kHlo) {
    graph_str = ir::DumpUtil::ToHlo(
        root_values, unique_device ? *unique_device : GetCurrentDevice());
  } else {
    XLA_ERROR() << "Invalid graph format: " << format;
  }
  ss << "\n## BEGIN_GRAPH\n" << graph_str << "\n## END_GRAPH\n\n";
  if (ir::Node::s_log_graph_changes_) {
    ss << ir::DumpUtil::GetGraphChangeLog(root_nodes) << "\n";
  }
  return ss.str();
}

void DebugUtil::SaveTensorsGraphInfo(const char* name,
                                     absl::Span<const XLATensor> tensors,
                                     const std::vector<size_t>* indices,
                                     GraphFormat format) {
  static const std::string save_file =
      xla::sys_util::GetEnvOrdinalPath("XLA_SAVE_TENSORS_FILE", "");
  if (!save_file.empty()) {
    static std::mutex lock;
    std::string info = GetTensorsGraphInfo(tensors, indices, format);
    std::lock_guard<std::mutex> guard(lock);
    std::ofstream graph_file(save_file, std::ios_base::app);
    graph_file << "[" << name << "]\n" << info << "\n";
  }
}

bool DebugUtil::ExperimentEnabled(const std::string& name) {
  static const absl::node_hash_set<std::string>* xset = LoadExperiments();
  return xset->find(name) != xset->end();
}

}  // namespace codira_xla
