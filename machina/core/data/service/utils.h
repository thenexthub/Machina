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
#ifndef MACHINA_CORE_DATA_SERVICE_UTILS_H_
#define MACHINA_CORE_DATA_SERVICE_UTILS_H_

#include <string>

#include "machina/core/data/service/common.pb.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/io/record_reader.h"
#include "machina/core/platform/env.h"

// Utilities shared between the dispatcher and worker servers.
namespace machina {
namespace data {

// Writes a dataset definition to the specified path. If the file already
// exists, it will be overwritten.
absl::Status WriteDatasetDef(const std::string& path,
                             const DatasetDef& dataset_def);

// Reads a dataset definition from specified path, and stores it in
// `dataset_def`. Returns NOT_FOUND if the path cannot be found.
absl::Status ReadDatasetDef(const std::string& path, DatasetDef& dataset_def);

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_DATA_SERVICE_UTILS_H_
