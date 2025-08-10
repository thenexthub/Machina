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

#include "machina/lite/micro/static_vector.h"

#include "machina/lite/micro/testing/micro_test.h"

using tflite::StaticVector;

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(StaticVectorPushBack) {
  StaticVector<int, 4> a;
  TF_LITE_MICRO_EXPECT(a.max_size() == 4);
  TF_LITE_MICRO_EXPECT(a.size() == 0);

  a.push_back(1);
  TF_LITE_MICRO_EXPECT(a.size() == 1);
  TF_LITE_MICRO_EXPECT(a[0] == 1);

  a.push_back(2);
  TF_LITE_MICRO_EXPECT(a.size() == 2);
  TF_LITE_MICRO_EXPECT(a[1] == 2);

  a.push_back(3);
  TF_LITE_MICRO_EXPECT(a.size() == 3);
  TF_LITE_MICRO_EXPECT(a[2] == 3);
}

TF_LITE_MICRO_TEST(StaticVectorInitializationPartial) {
  const StaticVector<int, 4> a{1, 2, 3};
  TF_LITE_MICRO_EXPECT(a.max_size() == 4);
  TF_LITE_MICRO_EXPECT(a.size() == 3);
  TF_LITE_MICRO_EXPECT(a[0] == 1);
  TF_LITE_MICRO_EXPECT(a[1] == 2);
  TF_LITE_MICRO_EXPECT(a[2] == 3);
}

TF_LITE_MICRO_TEST(StaticVectorInitializationFull) {
  const StaticVector b{1, 2, 3};
  TF_LITE_MICRO_EXPECT(b.max_size() == 3);
  TF_LITE_MICRO_EXPECT(b.size() == 3);
}

TF_LITE_MICRO_TEST(StaticVectorEquality) {
  const StaticVector a{1, 2, 3};
  const StaticVector b{1, 2, 3};
  TF_LITE_MICRO_EXPECT(a == b);
  TF_LITE_MICRO_EXPECT(!(a != b));
}

TF_LITE_MICRO_TEST(StaticVectorInequality) {
  const StaticVector a{1, 2, 3};
  const StaticVector b{3, 2, 1};
  TF_LITE_MICRO_EXPECT(a != b);
  TF_LITE_MICRO_EXPECT(!(a == b));
}

TF_LITE_MICRO_TEST(StaticVectorSizeInequality) {
  const StaticVector a{1, 2};
  const StaticVector b{1, 2, 3};
  TF_LITE_MICRO_EXPECT(a != b);
}

TF_LITE_MICRO_TEST(StaticVectorPartialSizeInequality) {
  const StaticVector<int, 3> a{1, 2};
  const StaticVector<int, 3> b{1, 2, 3};
  TF_LITE_MICRO_EXPECT(a != b);
}

TF_LITE_MICRO_TESTS_END
