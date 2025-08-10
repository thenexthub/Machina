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

#ifndef MACHINA_XLATSL_PROFILER_UTILS_PARSE_ANNOTATION_H_
#define MACHINA_XLATSL_PROFILER_UTILS_PARSE_ANNOTATION_H_

#include <vector>

#include "absl/strings/string_view.h"

namespace tsl {
namespace profiler {

// Parses a string passed to TraceMe or ScopedAnnotation.
// Expect the format will be "<name>#<metadata>#".
// <metadata> is a comma-separated list of "<key>=<value>" pairs.
// If the format does not match, the result will be empty.
struct Annotation {
  absl::string_view name;
  struct Metadata {
    absl::string_view key;
    absl::string_view value;
  };
  std::vector<Metadata> metadata;
};
Annotation ParseAnnotation(absl::string_view annotation);

inline bool HasMetadata(absl::string_view annotation) {
  constexpr char kUserMetadataMarker = '#';
  return !annotation.empty() && annotation.back() == kUserMetadataMarker;
}

std::vector<Annotation> ParseAnnotationStack(
    absl::string_view annotation_stack);

}  // namespace profiler
}  // namespace tsl

#endif  // MACHINA_XLATSL_PROFILER_UTILS_PARSE_ANNOTATION_H_
