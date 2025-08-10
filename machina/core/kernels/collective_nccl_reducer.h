/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef MACHINA_CORE_KERNELS_COLLECTIVE_NCCL_REDUCER_H_
#define MACHINA_CORE_KERNELS_COLLECTIVE_NCCL_REDUCER_H_

#include "machina/core/kernels/collective_nccl.h"

namespace machina {
#if GOOGLE_CUDA || MACHINA_USE_ROCM

class NcclReducer : public NcclBase {
 public:
  NcclReducer() : NcclBase(REDUCTION_COLLECTIVE, "NcclReduce") {}
  NcclReducer(CollectiveType type, const string& name) : NcclBase(type, name) {}
  ~NcclReducer() override = default;

  // Hands off all reduce to NcclManager.
  void Run(StatusCallback done) override;
};

class NcclReduceScatterer : public NcclReducer {
 public:
  NcclReduceScatterer()
      : NcclReducer(REDUCE_SCATTER_COLLECTIVE, "NcclReduceScatter") {}
  ~NcclReduceScatterer() override = default;
  // Uses same Run() as NcclReducer.
};

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_COLLECTIVE_NCCL_REDUCER_H_
