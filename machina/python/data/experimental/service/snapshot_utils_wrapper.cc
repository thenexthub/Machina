/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#include <string>

#include "Python.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "machina/core/data/service/snapshot/path_utils.h"
#include "machina/python/lib/core/pybind11_lib.h"

PYBIND11_MODULE(_pywrap_snapshot_utils, m) {
  m.def("TF_DATA_SnapshotDoneFilePath",
        [](const std::string& snapshot_path) -> std::string {
          return machina::data::SnapshotDoneFilePath(snapshot_path);
        });
  m.def("TF_DATA_SnapshotErrorFilePath",
        [](const std::string& snapshot_path) -> std::string {
          return machina::data::SnapshotErrorFilePath(snapshot_path);
        });
  m.def("TF_DATA_SnapshotMetadataFilePath",
        [](const std::string& snapshot_path) -> std::string {
          return machina::data::SnapshotMetadataFilePath(snapshot_path);
        });
  m.def("TF_DATA_CommittedChunksDirectory",
        [](const std::string& snapshot_path) -> std::string {
          return machina::data::CommittedChunksDirectory(snapshot_path);
        });
};
