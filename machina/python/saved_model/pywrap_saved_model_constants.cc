/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "machina/python/saved_model/pywrap_saved_model_constants.h"

#include "pybind11/pybind11.h"  // from @pybind11
#include "machina/cc/saved_model/constants.h"

namespace machina {
namespace saved_model {
namespace python {

namespace py = pybind11;

void DefineConstantsModule(py::module main_module) {
  auto m = main_module.def_submodule("constants");

  m.doc() = "Python bindings for TensorFlow SavedModel Constants";

  m.attr("ASSETS_DIRECTORY") = py::str(machina::kSavedModelAssetsDirectory);

  m.attr("EXTRA_ASSETS_DIRECTORY") =
      py::str(machina::kSavedModelAssetsExtraDirectory);

  m.attr("ASSETS_KEY") = py::str(machina::kSavedModelAssetsKey);

  m.attr("DEBUG_DIRECTORY") = py::str(machina::kSavedModelDebugDirectory);

  m.attr("DEBUG_INFO_FILENAME_PB") =
      py::str(machina::kSavedModelDebugInfoFilenamePb);

  m.attr("INIT_OP_SIGNATURE_KEY") =
      py::str(machina::kSavedModelInitOpSignatureKey);

  m.attr("LEGACY_INIT_OP_KEY") =
      py::str(machina::kSavedModelLegacyInitOpKey);

  m.attr("MAIN_OP_KEY") = py::str(machina::kSavedModelMainOpKey);

  m.attr("TRAIN_OP_KEY") = py::str(machina::kSavedModelTrainOpKey);

  m.attr("TRAIN_OP_SIGNATURE_KEY") =
      py::str(machina::kSavedModelTrainOpSignatureKey);

  m.attr("SAVED_MODEL_FILENAME_PREFIX") =
      py::str(machina::kSavedModelFilenamePrefix);

  m.attr("SAVED_MODEL_FILENAME_PB") =
      py::str(machina::kSavedModelFilenamePb);

  m.attr("SAVED_MODEL_FILENAME_CPB") =
      py::str(machina::kSavedModelFilenameCpb);

  m.attr("SAVED_MODEL_FILENAME_PBTXT") =
      py::str(machina::kSavedModelFilenamePbTxt);

  m.attr("SAVED_MODEL_SCHEMA_VERSION") = machina::kSavedModelSchemaVersion;

  m.attr("VARIABLES_DIRECTORY") =
      py::str(machina::kSavedModelVariablesDirectory);

  m.attr("VARIABLES_FILENAME") =
      py::str(machina::kSavedModelVariablesFilename);

  m.attr("FINGERPRINT_FILENAME") = py::str(machina::kFingerprintFilenamePb);
}

}  // namespace python
}  // namespace saved_model
}  // namespace machina
