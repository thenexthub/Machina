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

#ifndef MACHINA_CORE_UTIL_PADDING_H_
#define MACHINA_CORE_UTIL_PADDING_H_

// This file contains helper routines to deal with padding in various ops and
// kernels.

#include <string>
#include <vector>

#include "machina/core/lib/core/status.h"
#include "machina/core/util/tensor_format.h"

namespace machina {

class NodeDef;

// Padding: the padding we apply to the input tensor along the rows and columns
// dimensions. This is usually used to make sure that the spatial dimensions do
// not shrink when we progress with convolutions. Three types of padding are
// supported:
//   VALID: No padding is carried out.
//   SAME: The pad value is computed so that the output will have the same
//         dimensions as the input.
//   EXPLICIT: The user specifies the pad values in the explicit_paddings
//             attribute.
// The padded area is typically zero-filled. For pooling ops, the padded area is
// instead ignored. For max pool, this is equivalent to padding with -infinity.
enum Padding {
  VALID = 1,     // No padding.
  SAME = 2,      // Input and output layers have the same size.
  EXPLICIT = 3,  // Padding is explicitly specified
};

// Returns an error if the padding attributes are invalid.
absl::Status CheckValidPadding(Padding padding_type,
                               const std::vector<int64_t>& explicit_paddings,
                               int num_dims, TensorFormat data_format);

// Return the string containing the list of valid padding types, that can be
// used as an Attr() in REGISTER_OP.
std::string GetPaddingAttrString();

// Like GetPaddingAttrString(), but also includes EXPLICIT.
std::string GetPaddingAttrStringWithExplicit();

std::string GetExplicitPaddingsAttrString();

// Sets padding value based on the given string padding value.
absl::Status GetPaddingFromString(absl::string_view str_value, Padding* value);

}  // end namespace machina

#endif  // MACHINA_CORE_UTIL_PADDING_H_
