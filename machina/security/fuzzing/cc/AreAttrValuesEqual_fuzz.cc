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
#include <cassert>
#include <string>

#include "fuzztest/fuzztest.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/attr_value_util.h"

// This is a fuzzer for AreAttrValuesEqual.

namespace {

// A few helpers to construct AttrValue protos.
template <typename T>
machina::AttrValue createAttrValue(T value) {
  machina::AttrValue ret;
  SetAttrValue(value, &ret);
  return ret;
}

// A helper to do the comparison asserts.
template <typename T>
void compareValues(T value, T value_2) {
  const machina::AttrValue proto = createAttrValue(value);
  const machina::AttrValue proto_same = createAttrValue(value);
  const machina::AttrValue proto2 = createAttrValue(value_2);

  // Assert that AreAttrValuesEqual is true with or without allow false
  // negatives.
  assert(machina::AreAttrValuesEqual(proto, proto_same,
                                        /*allow_false_negatives=*/false));
  assert(machina::AreAttrValuesEqual(proto, proto_same,
                                        /*allow_false_negatives=*/true));
  // Assert that AreAttrValuesEqual are same with or without allow false
  // negatives.
  assert(machina::AreAttrValuesEqual(proto, proto2,
                                        /*allow_false_negatives=*/false) ==
         machina::AreAttrValuesEqual(proto, proto2,
                                        /*allow_false_negatives=*/true));
}

void FuzzTest(const int i, const int j, const float u, const float v,
              const std::string s1, const std::string s2) {
  compareValues(i, j);
  compareValues(u, v);
  compareValues(s1, s2);
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
    .WithDomains(fuzztest::InRange(1, 100), fuzztest::InRange(1, 1000),
                 fuzztest::InRange(1.0f, 1000.0f),
                 fuzztest::InRange(1.0f, 1000.0f),
                 fuzztest::Arbitrary<std::string>(),
                 fuzztest::Arbitrary<std::string>());

}  // namespace
