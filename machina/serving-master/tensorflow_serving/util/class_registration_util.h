/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#ifndef MACHINA_SERVING_UTIL_CLASS_REGISTRATION_UTIL_H_
#define MACHINA_SERVING_UTIL_CLASS_REGISTRATION_UTIL_H_

#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/strings/str_util.h"

namespace machina {
namespace serving {

// Parses a url whose final '/' is followed by a proto type name, e.g.
// "type.googleapis.com/some_namespace.some_proto_type_name".
// Returns Status::OK() iff parsing succeeded.
Status ParseUrlForAnyType(const string& type_url, string* const full_type_name);

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_UTIL_CLASS_REGISTRATION_UTIL_H_
