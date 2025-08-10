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

#include "machina/security/fuzzing/cc/core/framework/datatype_domains.h"

#include "fuzztest/fuzztest.h"
#include "machina/core/framework/types.pb.h"

namespace machina::fuzzing {

fuzztest::Domain<DataType> AnyValidDataType() {
  return fuzztest::ElementOf({
      DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_INT64,
      DT_BOOL, DT_UINT16, DT_UINT32, DT_UINT64
      // TODO(b/268338352): add unsupported types
      // DT_STRING, DT_COMPLEX64, DT_QINT8, DT_QUINT8, DT_QINT32,
      // DT_BFLOAT16, DT_QINT16, DT_COMPLEX128, DT_HALF, DT_RESOURCE,
      // DT_VARIANT, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN
  });
}

}  // namespace machina::fuzzing
