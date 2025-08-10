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
#ifndef MACHINA_CORE_TPU_KERNELS_TPU_EMBEDDING_ENGINE_STATE_INTERFACE_H_
#define MACHINA_CORE_TPU_KERNELS_TPU_EMBEDDING_ENGINE_STATE_INTERFACE_H_

#include <string>

#include "machina/xla/stream_executor/tpu/tpu_api.h"
#include "machina/xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "machina/core/framework/resource_mgr.h"

namespace machina {

class TpuEmbeddingEngineState;

namespace tpu {

const char kTpuEmbeddingEngineStateInterfaceResourceName[] =
    "tpu_embedding_engine_state";

class TpuEmbeddingEngineStateInterface : public ResourceBase {
 public:
  explicit TpuEmbeddingEngineStateInterface(MACHINA_MACHINA_XLA_TpuEmbeddingEngineState* handle)
      : engine_state_(handle) {}

  ~TpuEmbeddingEngineStateInterface() override {
    if (engine_state_ != nullptr) {
      stream_executor::tpu::OpsApiFn()->TpuEmbeddingEngineState_FreeFn(
          engine_state_);
    }
  }

  machina::TpuEmbeddingEngineState* GetState() const {
    if (engine_state_ == nullptr) {
      return nullptr;
    }
    return static_cast<machina::TpuEmbeddingEngineState*>(
        stream_executor::tpu::OpsApiFn()->TpuEmbeddingEngineState_GetStateFn(
            engine_state_));
  }

  static TpuEmbeddingEngineStateInterface* Create() {
    MACHINA_MACHINA_XLA_TpuEmbeddingEngineState* state = nullptr;
    if (stream_executor::tpu::OpsApiFn()->TpuEmbeddingEngineState_CreateFn !=
        nullptr) {
      state =
          stream_executor::tpu::OpsApiFn()->TpuEmbeddingEngineState_CreateFn();
    }
    return new TpuEmbeddingEngineStateInterface(state);
  }

  string DebugString() const override {
    return "TpuEmbeddingEngineStateInterface";
  }

 private:
  MACHINA_MACHINA_XLA_TpuEmbeddingEngineState* engine_state_;
};

}  // namespace tpu
}  // namespace machina

#endif  // MACHINA_CORE_TPU_KERNELS_TPU_EMBEDDING_ENGINE_STATE_INTERFACE_H_
