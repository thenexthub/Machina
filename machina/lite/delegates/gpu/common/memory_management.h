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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_H_

#include <stddef.h>

#include <vector>

#include "absl/memory/memory.h"
#include "machina/lite/delegates/gpu/common/memory_management/equality_assignment.h"
#include "machina/lite/delegates/gpu/common/memory_management/naive_assignment.h"
#include "machina/lite/delegates/gpu/common/memory_management/types.h"
#include "machina/lite/delegates/gpu/common/shape.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

using TaskId = size_t;

// Converts given assignment of tensors to shared objects to the assignment of
// the same tensors to offsets in continuous memory block.
OffsetsAssignment ObjectsToOffsets(
    const ObjectsAssignment<size_t>& obj_assignment);

enum class MemoryStrategy {
  // Naive strategy is to allocate each object separately.
  // Can be useful for debugging to see all intermediate outputs.
  NAIVE,

  // Equality strategy allows to reuse the same part of memory for several
  // tensors with the same size, but non-intersecting usage intervals.
  EQUALITY,

  // Greedy strategy uses greedy algorithm, iterating through all the tensors in
  // order of their first_task, to reuse memory from tensors, that
  // won't be used anymore, for new ones.
  GREEDY_IN_ORDER,

  // Greedy by size strategy uses greedy algorithm, iterating through all the
  // tasks in non-increasing of their breadth, and calculating allocations for
  // tensors used in these tasks. By breadth of the task we understand sum of
  // sizes of all tensors in its TaskProfile.
  GREEDY_BY_BREADTH,

  // Greedy by size strategy uses greedy algorithm, iterating through all the
  // tensors in non-increasing of their size, to reuse memory from tensors, that
  // won't be used anymore, for new ones.
  GREEDY_BY_SIZE,

  // Choose greedy strategy from several fast algorithms, that provides best
  // memory allocation for the given usage records.
  GREEDY_BEST,

  // Mincostflow strategy consists of building auxiliary flow graph and solving
  // the minimum-cost flow problem in it. In the end edges with zero residual
  // capacity determine assignment of shared objects to tensors.
  MINCOSTFLOW,
};

// Chooses greedy algorithm with the lowest memory consumption for given usage
// records and returns corresponding shared objects assignment.
absl::Status BestGreedy(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment);

// Calculates the assignment of shared objects to given tensors, including
// objects' sizes. Below there are specializations for different types, that
// support more memory strategies.
// If reallocation_graph is provided, assignment of shared objects support
// parallel order of operation execution, but memory consumption in this case
// can be larger. Currently only GREEDY_IN_ORDER strategy can use this
// reallocation_graph.
template <typename TensorSizeT>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<TensorSizeT>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<TensorSizeT>* assignment,
    const UsageGraph* reallocation_graph = nullptr) {
  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignment(usage_records, assignment);
    default:
      return absl::InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return absl::OkStatus();
}

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<size_t>* assignment,
    const UsageGraph* reallocation_graph);

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<BHWC>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<BHWC>* assignment,
    const UsageGraph* reallocation_graph);

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<uint2>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<uint2>* assignment,
    const UsageGraph* reallocation_graph);

template <>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<uint3>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<uint3>* assignment,
    const UsageGraph* reallocation_graph);

// Calculates the assignment of tensors to offsets, considering those tensors
// are going to be allocated in one continuous memory block.
absl::Status AssignOffsetsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    const MemoryStrategy& strategy, OffsetsAssignment* assignment,
    size_t base_addr_align_bytes = 1,
    const UsageGraph* reallocation_graph = nullptr);

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_H_
