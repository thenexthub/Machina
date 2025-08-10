/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_PYTHON_LIB_CORE_PYBIND11_ABSL_H_
#define MACHINA_PYTHON_LIB_CORE_PYBIND11_ABSL_H_

#include "pybind11/pybind11.h"  // from @pybind11
#include "machina/core/platform/stringpiece.h"

#ifndef ABSL_USES_STD_STRING_VIEW

namespace pybind11 {
namespace detail {

// Convert between machina::StringPiece (aka absl::string_view) and Python.
//
// pybind11 supports std::string_view, and absl::string_view is meant to be a
// drop-in replacement for std::string_view, so we can just use the built in
// implementation.
template <>
struct type_caster<machina::StringPiece>
    : string_caster<machina::StringPiece, true> {};

}  // namespace detail
}  // namespace pybind11

#endif  // ABSL_USES_STD_STRING_VIEW
#endif  // MACHINA_PYTHON_LIB_CORE_PYBIND11_ABSL_H_
