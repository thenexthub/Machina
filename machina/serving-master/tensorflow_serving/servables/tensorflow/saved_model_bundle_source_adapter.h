/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_BUNDLE_SOURCE_ADAPTER_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_BUNDLE_SOURCE_ADAPTER_H_

#include "machina/cc/saved_model/loader.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/macros.h"
#include "machina_serving/core/loader.h"
#include "machina_serving/core/simple_loader.h"
#include "machina_serving/core/source_adapter.h"
#include "machina_serving/core/storage_path.h"
#include "machina_serving/servables/machina/saved_model_bundle_factory.h"
#include "machina_serving/servables/machina/saved_model_bundle_source_adapter.pb.h"

namespace machina {
namespace serving {

// A SourceAdapter that creates SavedModelBundle Loaders from SavedModel paths.
// It keeps a SavedModelBundleFactory as its state, which may house a batch
// scheduler that is shared across all of the SavedModel bundles it emits.
class SavedModelBundleSourceAdapter final
    : public UnarySourceAdapter<StoragePath, std::unique_ptr<Loader>> {
 public:
  static Status Create(const SavedModelBundleSourceAdapterConfig& config,
                       std::unique_ptr<SavedModelBundleSourceAdapter>* adapter);

  ~SavedModelBundleSourceAdapter() override;

 private:
  friend class SavedModelBundleSourceAdapterCreator;

  explicit SavedModelBundleSourceAdapter(
      std::unique_ptr<SavedModelBundleFactory> bundle_factory);

  SimpleLoader<SavedModelBundle>::CreatorVariant GetServableCreator(
      std::shared_ptr<SavedModelBundleFactory> bundle_factory,
      const StoragePath& path) const;

  Status Convert(const StoragePath& path,
                 std::unique_ptr<Loader>* loader) override;

  // We use a shared ptr to share ownership with Loaders we emit, in case they
  // outlive this object.
  std::shared_ptr<SavedModelBundleFactory> bundle_factory_;

  TF_DISALLOW_COPY_AND_ASSIGN(SavedModelBundleSourceAdapter);
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_SAVED_MODEL_BUNDLE_SOURCE_ADAPTER_H_
