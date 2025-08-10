/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#ifndef MACHINA_C_EXPERIMENTAL_PLUGGABLE_PROFILER_PLUGGABLE_PROFILER_INTERNAL_H_
#define MACHINA_C_EXPERIMENTAL_PLUGGABLE_PROFILER_PLUGGABLE_PROFILER_INTERNAL_H_
#include "machina/c/experimental/pluggable_profiler/pluggable_profiler.h"
#include "machina/c/tf_status_helper.h"
#include "machina/core/platform/status.h"
#include "machina/core/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace machina {
namespace profiler {

// Plugin initialization function that a device plugin must define. Returns
// a TF_Status output specifying whether the initialization is successful.
using TFInitProfilerFn = void (*)(TF_ProfilerRegistrationParams* const,
                                  TF_Status* const);

// Registers plugin's profiler to TensorFlow's profiler registry.
absl::Status InitPluginProfiler(TFInitProfilerFn init_fn);

}  // namespace profiler
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_PLUGGABLE_PROFILER_PLUGGABLE_PROFILER_INTERNAL_H_
