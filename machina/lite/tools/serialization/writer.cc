/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
// Just does a read/write loop of tflite file format using the interpreter as
// an intermediate.
//
// Usage:
//   writer <input tflite> <output tflite>

#include <cstdio>
#include <memory>

#include "machina/lite/core/interpreter_builder.h"
#include "machina/lite/core/kernels/register.h"
#include "machina/lite/interpreter.h"
#include "machina/lite/model_builder.h"
#include "machina/lite/tools/serialization/writer_lib.h"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
    return 1;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(argv[1]);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates
      builtin_op_resolver;
  tflite::InterpreterBuilder(*model, builtin_op_resolver)(&interpreter);
  tflite::ModelWriter writer(interpreter.get());
  writer.Write(argv[2]);

  return 0;
}
