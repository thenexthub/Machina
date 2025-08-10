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
#ifndef MACHINA_CORE_TFRT_IFRT_CHECKPOINT_LOADER_H_
#define MACHINA_CORE_TFRT_IFRT_CHECKPOINT_LOADER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "machina/core/framework/types.pb.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/core/tfrt/fallback/fallback_state.h"
#include "machina/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "machina/core/tfrt/mlrt/bytecode/bytecode.h"
#include "machina/core/tfrt/mlrt/kernel/context.h"
#include "machina/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace machina {
namespace ifrt_serving {

// TODO(b/352551302) Move the unit test in ifrt_ops_kernel for restore to test
// this class's APIs.
// Implement the `CheckpointLoaderInterface` by using RestoreV2.
class CheckpointLoader {
 public:
  struct PrepareRestoreArgs {
    mlir::MLIRContext* context;
    machina::MetaGraphDef meta_graph_def;
    tfrt_stub::FallbackState* fallback_state;
    std::string saved_model_dir;
    bool run_placer_grappler_on_functions;
  };

  explicit CheckpointLoader(
      IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry,
      tfrt::ConcurrentWorkQueue* checkpoint_loader_work_queue,
      bool use_async_restore = true)
      : ifrt_restore_tensor_registry_(ifrt_restore_tensor_registry),
        checkpoint_loader_work_queue_(checkpoint_loader_work_queue),
        use_async_restore_(use_async_restore) {}
  virtual ~CheckpointLoader() = default;

  // Called before `Load` to do some preparation work.
  virtual absl::Status PrepareRestore(const PrepareRestoreArgs& args);

  // Load the checkpoint. This API is designed to be compatible with the
  // `tf_mlrt.ifrt_restore_variable` kernel.
  virtual absl::Status Load(
      const machina::tfrt_stub::FallbackTensor& prefix,
      const std::vector<machina::tfrt_stub::FallbackTensor>& var_handles,
      const machina::tfrt_stub::FallbackTensor& tensor_names,
      const machina::tfrt_stub::FallbackTensor& shape_and_slices,
      absl::Span<const machina::DataType> restored_dtypes,
      const std::vector<bool>& truncate_in_cast, tf_mlrt::Context& context);

 protected:
  IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry_;
  tfrt::ConcurrentWorkQueue* checkpoint_loader_work_queue_;
  bool use_async_restore_ = true;
};

}  // namespace ifrt_serving
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_IFRT_CHECKPOINT_LOADER_H_
