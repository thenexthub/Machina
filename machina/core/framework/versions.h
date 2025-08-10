/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_CORE_FRAMEWORK_VERSIONS_H_
#define MACHINA_CORE_FRAMEWORK_VERSIONS_H_

#include "machina/core/lib/core/status.h"

namespace machina {

class VersionDef;

// Check whether data with the given versions is compatible with the given
// consumer and min producer.  upper_name and lower_name are used to form
// error messages upon failure.  Example usage:
//
//   #include "machina/core/public/version.h"
//
//   TF_RETURN_IF_ERROR(CheckVersions(versions, TF_GRAPH_DEF_VERSION,
//                                    TF_GRAPH_DEF_VERSION_MIN_PRODUCER,
//                                    "GraphDef", "graph"));
absl::Status CheckVersions(const VersionDef& versions, int consumer,
                           int min_producer, const char* upper_name,
                           const char* lower_name);

}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_VERSIONS_H_
