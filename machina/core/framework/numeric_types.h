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

#ifndef MACHINA_CORE_FRAMEWORK_NUMERIC_TYPES_H_
#define MACHINA_CORE_FRAMEWORK_NUMERIC_TYPES_H_

#include <complex>

// clang-format off
// This include order is required to avoid instantiating templates
// quantized types in the Eigen namespace before their specialization.
#include "machina/xla/tsl/framework/numeric_types.h"
#include "machina/core/platform/types.h"
// clang-format on

namespace machina {

// NOLINTBEGIN(misc-unused-using-decls)
using tsl::complex128;
using tsl::complex64;

// We use Eigen's QInt implementations for our quantized int types.
using tsl::qint16;
using tsl::qint32;
using tsl::qint8;
using tsl::quint16;
using tsl::quint8;
// NOLINTEND(misc-unused-using-decls)

}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_NUMERIC_TYPES_H_
