/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#ifndef MACHINA_LITE_TOOLS_BENCHMARK_BENCHMARK_PARAMS_H_
#define MACHINA_LITE_TOOLS_BENCHMARK_BENCHMARK_PARAMS_H_
#include "machina/lite/tools/tool_params.h"

namespace tflite {
namespace benchmark {
using BenchmarkParam = tflite::tools::ToolParam;
using BenchmarkParams = tflite::tools::ToolParams;

// To be used in BenchmarkModel::LogParams() and its overrides as we assume
// logging the parameters defined in BenchmarkModel as 'params_'.
#define LOG_BENCHMARK_PARAM(type, name, description, verbose) \
  LOG_TOOL_PARAM(params_, type, name, description, verbose)
}  // namespace benchmark
}  // namespace tflite
#endif  // MACHINA_LITE_TOOLS_BENCHMARK_BENCHMARK_PARAMS_H_
