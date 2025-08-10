/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/lite/mutable_op_resolver_utils.h"

#include <stddef.h>

#include <gtest/gtest.h>
#include "machina/lite/builtin_ops.h"
#include "machina/lite/c/c_api.h"
#include "machina/lite/c/c_api_opaque.h"
#include "machina/lite/c/common.h"
#include "machina/lite/c/common_internal.h"
#include "machina/lite/mutable_op_resolver.h"
#include "machina/lite/schema/schema_generated.h"
#include "machina/lite/test_util.h"

namespace tflite {
namespace {

// We need some dummy functions to identify the registrations.
TfLiteStatus DummyInvoke(void* user_data, TfLiteOpaqueContext* context,
                         TfLiteOpaqueNode* node) {
  return kTfLiteOk;
}
TfLiteStatus DummyPrepare(void* user_data, TfLiteOpaqueContext* context,
                          TfLiteOpaqueNode* node) {
  return kTfLiteOk;
}

TfLiteOperator* GetDummyRegistration() {
  static TfLiteOperator* registration = []() {
    auto* op = TfLiteOperatorCreate(kTfLiteBuiltinCustom, "dummy",
                                    /*version=*/1, /*user_data=*/nullptr);
    TfLiteOperatorSetPrepareWithData(op, DummyPrepare);
    TfLiteOperatorSetInvokeWithData(op, DummyInvoke);
    return op;
  }();
  return registration;
}

TfLiteOperator* GetAdditionOpRegistration() {
  static TfLiteOperator* registration = []() {
    auto* r = TfLiteOperatorCreate(kTfLiteBuiltinAdd, /*custom_name=*/nullptr,
                                   /*version=*/1, /*user_data=*/nullptr);
    TfLiteOperatorSetInvokeWithData(r, DummyInvoke);
    return r;
  }();
  return registration;
}

using MutableOpResolverTest = tflite::testing::Test;

TEST_F(MutableOpResolverTest, FindOp) {
  MutableOpResolver resolver;
  AddOp(&resolver, GetAdditionOpRegistration());

  const TfLiteRegistration* found_registration =
      resolver.FindOp(BuiltinOperator_ADD, 1);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->registration_external->invoke_with_data ==
              DummyInvoke);
  EXPECT_EQ(
      TfLiteOperatorGetBuiltInCode(found_registration->registration_external),
      kTfLiteBuiltinAdd);
  EXPECT_EQ(TfLiteOperatorGetVersion(found_registration->registration_external),
            1);
  EXPECT_EQ(found_registration->builtin_code, BuiltinOperator_ADD);
  EXPECT_EQ(found_registration->version, 1);
}

TEST_F(MutableOpResolverTest, FindMissingOp) {
  MutableOpResolver resolver;
  AddOp(&resolver, GetAdditionOpRegistration());

  const TfLiteRegistration* found_registration =
      resolver.FindOp(BuiltinOperator_CONV_2D, 1);
  EXPECT_EQ(found_registration, nullptr);
}

TEST_F(MutableOpResolverTest, RegisterOpWithSingleVersion) {
  MutableOpResolver resolver;
  // The kernel supports version 2 only
  AddOp(&resolver, GetAdditionOpRegistration(), 2, 2);

  const TfLiteRegistration* found_registration;

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 1);
  ASSERT_EQ(found_registration, nullptr);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 2);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->registration_external->invoke_with_data ==
              DummyInvoke);
  EXPECT_EQ(found_registration->version, 2);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 3);
  ASSERT_EQ(found_registration, nullptr);
}

TEST_F(MutableOpResolverTest, RegisterOpWithMultipleVersions) {
  MutableOpResolver resolver;
  // The kernel supports version 2 and 3
  AddOp(&resolver, GetAdditionOpRegistration(), 2, 3);

  const TfLiteRegistration* found_registration;

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 2);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->registration_external->invoke_with_data ==
              DummyInvoke);
  EXPECT_EQ(found_registration->version, 2);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 3);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->registration_external->invoke_with_data ==
              DummyInvoke);
  EXPECT_EQ(found_registration->version, 3);
}

TEST_F(MutableOpResolverTest, FindOpWithUnsupportedVersions) {
  MutableOpResolver resolver;
  // The kernel supports version 2 and 3
  AddOp(&resolver, GetAdditionOpRegistration(), 2, 3);

  const TfLiteRegistration* found_registration;

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 1);
  EXPECT_EQ(found_registration, nullptr);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 4);
  EXPECT_EQ(found_registration, nullptr);
}

TEST_F(MutableOpResolverTest, FindCustomOp) {
  MutableOpResolver resolver;
  AddOp(&resolver, GetDummyRegistration());

  const TfLiteRegistration* found_registration = resolver.FindOp("dummy", 1);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_EQ(found_registration->builtin_code, BuiltinOperator_CUSTOM);
  EXPECT_TRUE(found_registration->registration_external->invoke_with_data ==
              DummyInvoke);
  EXPECT_EQ(found_registration->version, 1);
}

TEST_F(MutableOpResolverTest, FindMissingCustomOp) {
  MutableOpResolver resolver;
  AddOp(&resolver, GetDummyRegistration());

  const TfLiteRegistration* found_registration = resolver.FindOp("whatever", 1);
  EXPECT_EQ(found_registration, nullptr);
}

TEST_F(MutableOpResolverTest, FindCustomOpWithUnsupportedVersion) {
  MutableOpResolver resolver;
  AddOp(&resolver, GetDummyRegistration());

  const TfLiteRegistration* found_registration = resolver.FindOp("dummy", 2);
  EXPECT_EQ(found_registration, nullptr);
}

}  // namespace
}  // namespace tflite
