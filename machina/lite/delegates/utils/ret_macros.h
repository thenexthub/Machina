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
#ifndef MACHINA_LITE_DELEGATES_UTILS_RET_MACROS_H_
#define MACHINA_LITE_DELEGATES_UTILS_RET_MACROS_H_

#include <cstddef>
#include <cstdlib>

#include "machina/lite/c/c_api_types.h"
#include "machina/lite/minimal_logging.h"

// Evaluate an expression whose type is std::optional<T>. If it returns an
// instance that contains a value, then initialize the declaration with that
// value; otherwise, return the specified error value.
#define TFLITE_ASSIGN_OR_RETURN(declaration, expr, err)                \
  TFLITE_ASSIGN_OR_RETURN_IMPL(                                        \
      TFLITE_ASSIGN_OR_RETURN_MACROS_CONCAT_NAME(_maybe, __COUNTER__), \
      declaration, expr, err);

#define TFLITE_ASSIGN_OR_RETURN_IMPL(maybe, declaration, expr, err) \
  auto maybe = (expr);                                              \
  if (!maybe.has_value()) return (err);                             \
  declaration = *std::move(maybe)

#define TFLITE_ASSIGN_OR_RETURN_MACROS_CONCAT_NAME(x, y) \
  TFLITE_ASSIGN_OR_RETURN_MACROS_CONCAT_IMPL(x, y)
#define TFLITE_ASSIGN_OR_RETURN_MACROS_CONCAT_IMPL(x, y) x##y

// If the specified condition is false, log the specified message and return the
// specified error value.
#define TFLITE_RET_CHECK(c, m, r) \
  TFLITE_RET_CHECK_IMPL(c, m, r, __FILE__, __LINE__)

#define TFLITE_RET_CHECK_IMPL(c, m, r, file, line)                         \
  do {                                                                     \
    if (!(c)) {                                                            \
      ::tflite::delegates::utils::TfLiteCheckLog("TFLITE_RET_CHECK", file, \
                                                 line, #c, m);             \
      return (r);                                                          \
    }                                                                      \
  } while (false)

// If the specified condition is false, log the specified message and return
// kTfLiteDelegateError.
#define TFLITE_RET_CHECK_STATUS(c, m) \
  TFLITE_RET_CHECK(c, m, kTfLiteDelegateError)

// If the specified condition is false, log the specified message and abort().
#define TFLITE_ABORT_CHECK(c, m) \
  TFLITE_ABORT_CHECK_IMPL(c, m, __FILE__, __LINE__)

#define TFLITE_ABORT_CHECK_IMPL(c, m, file, line)                            \
  do {                                                                       \
    if (!(c)) {                                                              \
      ::tflite::delegates::utils::TfLiteCheckLog("TFLITE_ABORT_CHECK", file, \
                                                 line, #c, m);               \
      std::abort();                                                          \
    }                                                                        \
  } while (false)

namespace tflite::delegates::utils {
inline void TfLiteCheckLog(const char* kind, const char* file, std::size_t line,
                           const char* cond, const char* message) {
  TFLITE_LOG_PROD(::tflite::TFLITE_LOG_ERROR, "%s failure (%s:%zu) %s \"%s\"",
                  kind, file, line, cond, message);
}
}  // namespace tflite::delegates::utils

#endif  // MACHINA_LITE_DELEGATES_UTILS_RET_MACROS_H_
