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

#ifndef MACHINA_CORE_TRANSFORMS_PASS_REGISTRATION_H_
#define MACHINA_CORE_TRANSFORMS_PASS_REGISTRATION_H_

#include <memory>

#include "machina/core/transforms/cf_sink/pass.h"
#include "machina/core/transforms/consolidate_attrs/pass.h"
#include "machina/core/transforms/const_dedupe_hoist/pass.h"
#include "machina/core/transforms/constant_folding/pass.h"
#include "machina/core/transforms/cse/pass.h"
#include "machina/core/transforms/drop_unregistered_attribute/pass.h"
#include "machina/core/transforms/eliminate_passthrough_iter_args/pass.h"
#include "machina/core/transforms/func_to_graph/pass.h"
#include "machina/core/transforms/functional_to_region/pass.h"
#include "machina/core/transforms/graph_compactor/pass.h"
#include "machina/core/transforms/graph_to_func/pass.h"
#include "machina/core/transforms/legacy_call/pass.h"
#include "machina/core/transforms/region_to_functional/pass.h"
#include "machina/core/transforms/remapper/pass.h"
#include "machina/core/transforms/shape_inference/pass.h"
#include "machina/core/transforms/toposort/pass.h"

namespace mlir {
namespace tfg {

// Generate the code for registering passes for command-line parsing.
#define GEN_PASS_REGISTRATION
#include "machina/core/transforms/passes.h.inc"

}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_CORE_TRANSFORMS_PASS_REGISTRATION_H_
