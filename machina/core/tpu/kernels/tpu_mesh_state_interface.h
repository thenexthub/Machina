/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#ifndef MACHINA_CORE_TPU_KERNELS_TPU_MESH_STATE_INTERFACE_H_
#define MACHINA_CORE_TPU_KERNELS_TPU_MESH_STATE_INTERFACE_H_

#include <string>

#include "machina/xla/stream_executor/tpu/tpu_api.h"
#include "machina/xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"

namespace machina {

class TpuMeshCommonState;

namespace tpu {

const char kTpuMeshStateInterfaceResourceName[] = "tpu_mesh_common_state";

class TpuMeshStateInterface : public machina::ResourceBase {
 public:
  explicit TpuMeshStateInterface(MACHINA_MACHINA_XLA_TpuMeshState* handle)
      : mesh_state_(handle) {}

  ~TpuMeshStateInterface() override {
    if (mesh_state_ != nullptr) {
      stream_executor::tpu::OpsApiFn()->TpuMeshState_FreeFn(mesh_state_);
    }
  }

  static TpuMeshStateInterface* Create() {
    MACHINA_MACHINA_XLA_TpuMeshState* state = nullptr;
    if (stream_executor::tpu::OpsApiFn()->TpuMeshState_CreateFn != nullptr) {
      state = stream_executor::tpu::OpsApiFn()->TpuMeshState_CreateFn();
    }
    return new TpuMeshStateInterface(state);
  }

  const MACHINA_MACHINA_XLA_TpuMeshState* data() const { return mesh_state_; }

  machina::TpuMeshCommonState* mesh_common_state() const {
    if (mesh_state_ == nullptr) {
      return nullptr;
    }
    return static_cast<machina::TpuMeshCommonState*>(
        stream_executor::tpu::OpsApiFn()->TpuMeshState_MeshCommonStateFn(
            mesh_state_));
  }

  // Returns whether we should include the device assignment as a static field
  // to the TPU program. This also determines whether we should include the
  // device assignment as part of the compilation cache key.
  bool NeedsStaticDeviceAssignment(const TPUCompileMetadataProto& metadata,
                                   TpuCoreTypeEnum tpu_core_type) const {
    if (mesh_state_ == nullptr) {
      return false;
    }
    // Static device assignment enables XLA to perform certain optimization when
    // all cores are used in the replicated computation.
    return metadata.num_cores_per_replica() * metadata.num_replicas() ==
           stream_executor::tpu::OpsApiFn()->TpuTopology_AvailableCoreCountFn(
               mesh_state_, tpu_core_type);
  }

  string DebugString() const override { return "TpuMeshStateInterface"; }

 private:
  MACHINA_MACHINA_XLA_TpuMeshState* mesh_state_;
};

}  // namespace tpu
}  // namespace machina

#endif  // MACHINA_CORE_TPU_KERNELS_TPU_MESH_STATE_INTERFACE_H_
