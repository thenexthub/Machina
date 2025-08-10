/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
#include "machina/lite/c/c_api_opaque_internal.h"

#include <memory>

#include <gtest/gtest.h>
#include "machina/lite/builtin_ops.h"
#include "machina/lite/core/subgraph.h"
#include "machina/lite/interpreter.h"
#include "machina/lite/interpreter_builder.h"
#include "machina/lite/kernels/builtin_op_kernels.h"
#include "machina/lite/kernels/register.h"
#include "machina/lite/model_builder.h"

using tflite::FlatBufferModel;
using tflite::Interpreter;
using tflite::InterpreterBuilder;
using tflite::internal::CommonOpaqueConversionUtil;
using tflite::ops::builtin::BuiltinOpResolver;

TEST(ObtainRegistrationFromContext, ProducesValidResult) {
  BuiltinOpResolver op_resolver;
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(
      "machina/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  InterpreterBuilder builder(*model, op_resolver);
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  TfLiteContext* context = interpreter->primary_subgraph().context();
  const TfLiteRegistration* registration = tflite::ops::builtin::Register_ADD();

  TfLiteOperator* registration_external =
      CommonOpaqueConversionUtil::ObtainOperator(context, registration, 42);

  ASSERT_EQ(registration_external->builtin_code, kTfLiteBuiltinAdd);
  ASSERT_EQ(registration_external->version, registration->version);
  ASSERT_EQ(registration_external->custom_name, registration->custom_name);
  ASSERT_EQ(registration_external->node_index, 42);
}

TEST(ObtainRegistrationFromContext, CachingWorks) {
  BuiltinOpResolver op_resolver;
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(
      "machina/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  InterpreterBuilder builder(*model, op_resolver);
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  TfLiteContext* context = interpreter->primary_subgraph().context();
  const TfLiteRegistration* registration = tflite::ops::builtin::Register_ADD();

  // Call it twice, and verify that we get the same result back.
  TfLiteOperator* registration_external1 =
      CommonOpaqueConversionUtil::ObtainOperator(context, registration, 0);
  TfLiteOperator* registration_external2 =
      CommonOpaqueConversionUtil::ObtainOperator(context, registration, 1);
  ASSERT_EQ(registration_external1, registration_external2);
}
