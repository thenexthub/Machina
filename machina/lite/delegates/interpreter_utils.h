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

#ifndef MACHINA_LITE_DELEGATES_INTERPRETER_UTILS_H_
#define MACHINA_LITE_DELEGATES_INTERPRETER_UTILS_H_

#include "machina/lite/interpreter.h"

// Utility functions and classes for using delegates.

namespace tflite {
namespace delegates {
class InterpreterUtils {
 public:
  /// Invokes an interpreter with automatic fallback from delegation to CPU.
  ///
  /// If using the delegate fails, the delegate is automatically undone and an
  /// attempt made to return the interpreter to an invokable state.
  ///
  /// Allowing the fallback is suitable only if both of the following hold:
  /// - The caller is known not to cache pointers to tensor data across Invoke()
  ///   calls.
  /// - The model is not stateful (no variables, no LSTMs) or the state isn't
  ///   needed between batches.
  ///
  /// Returns one of the following three status codes:
  /// 1. kTfLiteOk: Success. Output is valid.
  /// 2. kTfLiteDelegateError: Delegate error but fallback succeeded. Output is
  /// valid.
  /// NOTE: This undoes all delegates previously applied to the Interpreter.
  /// 3. kTfLiteError: Unexpected/runtime failure. Output is invalid.
  /// WARNING: This is an experimental API and subject to change.
  static TfLiteStatus InvokeWithCPUFallback(Interpreter* interpreter);
};
}  // namespace delegates
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_INTERPRETER_UTILS_H_
