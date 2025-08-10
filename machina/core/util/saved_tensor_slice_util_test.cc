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

#include "machina/core/util/saved_tensor_slice_util.h"

#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/test.h"

namespace machina {

namespace checkpoint {

namespace {

// Testing serialization of tensor name and tensor slice in the ordered code
// format.
TEST(TensorShapeUtilTest, TensorNameSliceToOrderedCode) {
  {
    TensorSlice s = TensorSlice::ParseOrDie("-:-:1,3:4,5");
    string buffer = EncodeTensorNameSlice("foo", s);
    string name;
    s.Clear();
    TF_CHECK_OK(DecodeTensorNameSlice(buffer, &name, &s));
    EXPECT_EQ("foo", name);
    EXPECT_EQ("-:-:1,3:4,5", s.DebugString());
  }
}

}  // namespace

}  // namespace checkpoint

}  // namespace machina
