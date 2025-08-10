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
#ifndef MACHINA_CORE_DATA_SERVICE_DATASET_STORE_H_
#define MACHINA_CORE_DATA_SERVICE_DATASET_STORE_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "machina/core/data/service/dispatcher_state.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/io/record_reader.h"
#include "machina/core/lib/io/record_writer.h"
#include "machina/core/platform/env.h"

namespace machina {
namespace data {

// An interface for storing and getting dataset definitions.
class DatasetStore {
 public:
  virtual ~DatasetStore() = default;

  // Stores the given dataset under the given key. Overwrites a dataset if it
  // already exists.
  virtual absl::Status Put(const std::string& key,
                           const DatasetDef& dataset) = 0;
  // Gets the dataset for the given key, storing the dataset in `dataset_def`.
  virtual absl::Status Get(const std::string& key,
                           std::shared_ptr<const DatasetDef>& dataset_def) = 0;
};

// Dataset store which reads and writes datasets within a directory.
// The dataset with key `key` is stored at the path "datasets_dir/key".
class FileSystemDatasetStore : public DatasetStore {
 public:
  explicit FileSystemDatasetStore(const std::string& datasets_dir);
  FileSystemDatasetStore(const FileSystemDatasetStore&) = delete;
  FileSystemDatasetStore& operator=(const FileSystemDatasetStore&) = delete;

  absl::Status Put(const std::string& key, const DatasetDef& dataset) override;
  absl::Status Get(const std::string& key,
                   std::shared_ptr<const DatasetDef>& dataset_def) override;

 private:
  const std::string datasets_dir_;
};

// DatasetStore which stores all datasets in memory. This is useful when the
// dispatcher doesn't have a work directory configured.
class MemoryDatasetStore : public DatasetStore {
 public:
  MemoryDatasetStore() = default;
  MemoryDatasetStore(const MemoryDatasetStore&) = delete;
  MemoryDatasetStore& operator=(const MemoryDatasetStore&) = delete;

  absl::Status Put(const std::string& key, const DatasetDef& dataset) override;
  absl::Status Get(const std::string& key,
                   std::shared_ptr<const DatasetDef>& dataset_def) override;

 private:
  // Mapping from key to dataset definition.
  absl::flat_hash_map<std::string, std::shared_ptr<const DatasetDef>> datasets_;
};

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_DATA_SERVICE_DATASET_STORE_H_
