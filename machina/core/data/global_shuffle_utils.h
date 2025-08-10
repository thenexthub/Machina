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
#ifndef MACHINA_CORE_DATA_GLOBAL_SHUFFLE_UTILS_H_
#define MACHINA_CORE_DATA_GLOBAL_SHUFFLE_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "machina/core/framework/dataset.h"
#include "machina/core/framework/tensor.h"

namespace machina {
namespace data {

// Builds and selects the `IteratorContext` to use based on whether the dataset
// is globally shuffled.
//
// Example usage in `Iterator::GetNextInternal`:
//
// ```
// IteratorContextWithIndexMapper ctx_with_index_mapper(ctx, this);
// TF_RETURN_IF_ERROR(input_impl_->GetNext(
//     ctx_with_index_mapper.Get(), out_tensors, end_of_sequence));
// ctx_with_index_mapper.MergeCheckpoint();
// ```
//
// The iterator should also implement `GetIndexMapper` if it needs to customize
// the index mapping behavior.
class IteratorContextWithIndexMapper {
 public:
  // Caller keeps ownership of both pointers.
  explicit IteratorContextWithIndexMapper(IteratorContext* ctx,
                                          const IteratorBase* iterator);
  virtual ~IteratorContextWithIndexMapper() = default;
  IteratorContextWithIndexMapper(const IteratorContextWithIndexMapper&) =
      delete;
  IteratorContextWithIndexMapper& operator=(
      const IteratorContextWithIndexMapper&) = delete;

  IteratorContext* Get();
  void MergeCheckpoint();

 private:
  IteratorContext* ctx_;
  std::optional<IteratorContext> ctx_with_index_mapper_;
};

// For source datasets that support random access, this class adapts the dataset
// random access API to support globally shuffled iterators.
class GlobalShuffleIterator {
 public:
  // The dataset is expected to support random access by implementing the
  // absl::Status Get(int64_t index, std::vector<Tensor>* out_tensors) const.
  explicit GlobalShuffleIterator(const DatasetBase* dataset)
      : dataset_(dataset) {}

  // Returns the next shuffled element.
  // REQUIRES: ctx->index_mapper() != nullptr.
  absl::Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                       bool* end_of_sequence);

  absl::Status Save(const std::string& parent_iterator_prefix,
                    SerializationContext* ctx, IteratorStateWriter* writer);

  // Restores the element count.
  // REQUIRES: ctx->restored_element_count() != nullopt.
  absl::Status Restore(const std::string& parent_iterator_prefix,
                       IteratorContext* ctx, IteratorStateReader* reader);

 private:
  const DatasetBase* const dataset_;

  mutable absl::Mutex mu_;

  // Count of elements produced by this iterator when it runs in the random
  // access mode.
  int64_t element_count_ ABSL_GUARDED_BY(mu_) = 0;
};

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_DATA_GLOBAL_SHUFFLE_UTILS_H_
