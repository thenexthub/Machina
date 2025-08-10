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
#ifndef MACHINA_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
#define MACHINA_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_

#include <memory>
#include <stack>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "machina/xla/python/profiler/internal/python_hooks.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace machina {
namespace profiler {

using xla::profiler::PythonHooksOptions;  // NOLINT

using xla::profiler::PythonTraceEntry;  // NOLINT

using xla::profiler::PerThreadEvents;  // NOLINT

using xla::profiler::PythonHookContext;  // NOLINT

using xla::profiler::PythonHooks;  // NOLINT

}  // namespace profiler
}  // namespace machina

#endif  // MACHINA_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
