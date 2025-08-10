/* Copyright 2020 Google Inc. All Rights Reserved.

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

#include "machina_serving/servables/machina/machine_learning_metadata.h"

#include "machina/cc/saved_model/constants.h"
#include "machina/core/lib/io/path.h"
#include "machina/core/lib/monitoring/gauge.h"
#include "machina/core/lib/strings/strcat.h"

namespace machina {
namespace serving {

namespace {

static constexpr char kMLMDKeyFile[] = "mlmd_uuid";

auto* mlmd_map = monitoring::Gauge<string, 2>::New(
    "/machina/serving/mlmd_map",
    "Mapping for ML Metadata UUID to model_name and version.", "model_name",
    "version");

}  // namespace

void MaybePublishMLMDStreamz(const string& export_dir, const string& model_name,
                             int64_t version) {
  const string mlmd_path = machina::io::JoinPath(
      export_dir, machina::kSavedModelAssetsExtraDirectory, kMLMDKeyFile);
  if (tsl::Env::Default()->FileExists(mlmd_path).ok()) {
    string mlmd_key;
    auto status = ReadFileToString(tsl::Env::Default(), mlmd_path, &mlmd_key);
    if (!status.ok()) {
      LOG(WARNING) << "ML Metadata Key Found But couldn't be read.";
    } else {
      mlmd_map->GetCell(model_name, strings::StrCat(version))->Set(mlmd_key);
    }
  }
}

}  // namespace serving
}  // namespace machina
