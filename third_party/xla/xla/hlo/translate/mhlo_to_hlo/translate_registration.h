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

#ifndef MACHINA_XLAHLO_TRANSLATE_MHLO_TO_HLO_TRANSLATE_REGISTRATION_H_
#define MACHINA_XLAHLO_TRANSLATE_MHLO_TO_HLO_TRANSLATE_REGISTRATION_H_

#include "toolchain/Support/CommandLine.h"

// NOLINTNEXTLINE
toolchain::cl::opt<bool> emit_use_tuple_arg(
    "emit-use-tuple-args",
    toolchain::cl::desc(
        "Emit HLO modules using tuples as args for the entry computation"),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
toolchain::cl::opt<bool> emit_return_tuple(
    "emit-return-tuple",
    toolchain::cl::desc("Emit HLO modules with entry computations returning tuple"),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
toolchain::cl::opt<bool> with_layouts(
    "with-layouts",
    toolchain::cl::desc("Propagate layouts when translating MHLO->XLA HLO"),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
toolchain::cl::opt<bool> print_layouts(
    "print-layouts", toolchain::cl::desc("Print layouts in the generated HLO text"),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
toolchain::cl::opt<bool> print_large_constants(
    "print-large-constants",
    toolchain::cl::desc("Print large constants in the generated HLO text"),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
toolchain::cl::opt<bool> print_sugar(
    "print-sugar",
    toolchain::cl::desc(
        "Print async ops using syntactic sugar in the generated HLO text"),
    toolchain::cl::init(true));

// NOLINTNEXTLINE
toolchain::cl::opt<bool> via_builder(
    "via-builder", toolchain::cl::desc("Translate MHLO->XLA HLO via XLA Builder"),
    toolchain::cl::init(false));

#endif  // MACHINA_XLAHLO_TRANSLATE_MHLO_TO_HLO_TRANSLATE_REGISTRATION_H_
