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

#ifndef MACHINA_SECURITY_FUZZING_CC_CORE_FRAMEWORK_DATATYPE_DOMAINS_H_
#define MACHINA_SECURITY_FUZZING_CC_CORE_FRAMEWORK_DATATYPE_DOMAINS_H_

#include "fuzztest/fuzztest.h"
#include "machina/core/framework/types.pb.h"

namespace machina::fuzzing {

/// Returns a fuzztest domain of valid DataTypes to construct a Tensor
fuzztest::Domain<DataType> AnyValidDataType();

}  // namespace machina::fuzzing

#endif  // MACHINA_SECURITY_FUZZING_CC_CORE_FRAMEWORK_DATATYPE_DOMAINS_H_
