/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#include "machina_serving/servables/machina/simple_servers.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina_serving/config/file_system_storage_path_source.pb.h"
#include "machina_serving/core/aspired_versions_manager_builder.h"
#include "machina_serving/core/availability_preserving_policy.h"
#include "machina_serving/core/loader.h"
#include "machina_serving/core/source.h"
#include "machina_serving/core/source_adapter.h"
#include "machina_serving/core/storage_path.h"
#include "machina_serving/core/target.h"
#include "machina_serving/servables/machina/saved_model_bundle_source_adapter.h"
#include "machina_serving/servables/machina/saved_model_bundle_source_adapter.pb.h"
#include "machina_serving/sources/storage_path/file_system_storage_path_source.h"

namespace machina {
namespace serving {
namespace simple_servers {

namespace {

// Creates a Source<StoragePath> that monitors a filesystem's base_path for new
// directories. Upon finding these, it provides the target with the new version
// (a directory). The servable_name param simply allows this source to create
// all AspiredVersions for the target with the same servable_name.
absl::Status CreateStoragePathSource(
    const string& base_path, const string& servable_name,
    std::unique_ptr<Source<StoragePath>>* path_source) {
  FileSystemStoragePathSourceConfig config;
  config.set_file_system_poll_wait_seconds(1);
  auto* servable = config.add_servables();
  servable->set_servable_name(servable_name);
  servable->set_base_path(base_path);

  std::unique_ptr<FileSystemStoragePathSource> file_system_source;
  TF_RETURN_IF_ERROR(
      FileSystemStoragePathSource::Create(config, &file_system_source));

  *path_source = std::move(file_system_source);
  return absl::OkStatus();
}

// Creates a SavedModelBundle Source by adapting the underlying
// FileSystemStoragePathSource. These two are connected in the
// 'CreateSingleTFModelManagerFromBasePath' method, with the
// FileSystemStoragePathSource as the Source and the SavedModelBundleSource as
// the Target.
absl::Status CreateSavedModelBundleSource(
    std::unique_ptr<SavedModelBundleSourceAdapter>* source) {
  SavedModelBundleSourceAdapterConfig config;
  TF_RETURN_IF_ERROR(SavedModelBundleSourceAdapter::Create(config, source));

  return absl::OkStatus();
}

}  // namespace

absl::Status CreateSingleTFModelManagerFromBasePath(
    const string& base_path, std::unique_ptr<Manager>* const manager) {
  std::unique_ptr<SavedModelBundleSourceAdapter> bundle_source;
  TF_RETURN_IF_ERROR(CreateSavedModelBundleSource(&bundle_source));
  std::unique_ptr<Source<StoragePath>> path_source;
  TF_RETURN_IF_ERROR(
      CreateStoragePathSource(base_path, "default", &path_source));

  AspiredVersionsManagerBuilder::Options manager_options;
  manager_options.aspired_version_policy.reset(
      new AvailabilityPreservingPolicy);
  std::unique_ptr<AspiredVersionsManagerBuilder> builder;
  TF_CHECK_OK(AspiredVersionsManagerBuilder::Create(std::move(manager_options),
                                                    &builder));
  builder->AddSourceChain(std::move(path_source), std::move(bundle_source));
  *manager = builder->Build();

  return absl::OkStatus();
}

}  // namespace simple_servers
}  // namespace serving
}  // namespace machina
