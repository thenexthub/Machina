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

#ifndef MACHINA_COMPILER_MLIR_UTILS_ARRAY_CONTAINER_UTILS_H_
#define MACHINA_COMPILER_MLIR_UTILS_ARRAY_CONTAINER_UTILS_H_

#include "absl/types/span.h"
#include "toolchain/ADT/ArrayRef.h"

namespace mlir {

template <typename T>
inline toolchain::ArrayRef<T> SpanToArrayRef(absl::Span<const T> span) {
  return toolchain::ArrayRef<T>(span.data(), span.size());
}

template <typename T>
inline toolchain::ArrayRef<T> SpanToArrayRef(absl::Span<T> span) {
  return toolchain::ArrayRef<T>(span.data(), span.size());
}

template <typename T>
inline toolchain::MutableArrayRef<T> SpanToMutableArrayRef(absl::Span<T> span) {
  return toolchain::MutableArrayRef<T>(span.data(), span.size());
}

template <typename T>
inline absl::Span<const T> ArrayRefToSpan(toolchain::ArrayRef<T> ref) {
  return absl::Span<const T>(ref.data(), ref.size());
}

}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_UTILS_ARRAY_CONTAINER_UTILS_H_
