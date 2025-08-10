/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#ifndef XLA_CLIENT_TF_LOGGING_H_
#define XLA_CLIENT_TF_LOGGING_H_

#include <sstream>

#include "machina/compiler/xla/status.h"
#include "machina/core/platform/logging.h"

namespace xla {
namespace internal {

// It happens that Caffe defined the same exact Google macros, hiding the TF
// ones, and making log messages disappear.
// Unfortunately to get those back, we have to poke through the TF
// implementaiton of them.
#define TF_LOG(severity) _TF_LOG_##severity

#define TF_VLOG_IS_ON(lvl)                                                  \
  (([](int level, const char* fname) {                                      \
    static const bool vmodule_activated =                                   \
        ::machina::internal::LogMessage::VmoduleActivated(fname, level); \
    return vmodule_activated;                                               \
  })(lvl, __FILE__))

#define TF_VLOG(level)                                           \
  TF_PREDICT_TRUE(!TF_VLOG_IS_ON(level))                         \
  ? (void)0                                                      \
  : ::machina::internal::Voidifier() &                        \
          ::machina::internal::LogMessage(__FILE__, __LINE__, \
                                             machina::INFO)

struct ErrorSink : public std::basic_ostringstream<char> {};

class ErrorGenerator {
 public:
  ErrorGenerator(const char* file, int line) : file_(file), line_(line) {}

  // Use a dummy & operator as it has lower precedence WRT the streaming
  // operator, and hence allows collecting user error messages before we finally
  // throw.
  TF_ATTRIBUTE_NORETURN void operator&(
      const std::basic_ostream<char>& oss) const;

 private:
  const char* file_ = nullptr;
  int line_ = 0;
};

#define TF_ERROR_STREAM()                               \
  ::xla::internal::ErrorGenerator(__FILE__, __LINE__) & \
      ::xla::internal::ErrorSink()

#define TF_CHECK(condition)              \
  while (TF_PREDICT_FALSE(!(condition))) \
  TF_ERROR_STREAM() << "Check failed: " #condition " "

#define TF_CHECK_OP_LOG(name, op, val1, val2)                         \
  while (::machina::internal::CheckOpString _result{               \
      ::machina::internal::name##Impl(                             \
          ::machina::internal::GetReferenceableValue(val1),        \
          ::machina::internal::GetReferenceableValue(val2),        \
          #val1 " " #op " " #val2)})                                  \
  TF_ERROR_STREAM() << *(_result.str_)

#define TF_CHECK_OP(name, op, val1, val2) TF_CHECK_OP_LOG(name, op, val1, val2)

// TF_CHECK_EQ/NE/...
#define TF_CHECK_EQ(val1, val2) TF_CHECK_OP(Check_EQ, ==, val1, val2)
#define TF_CHECK_NE(val1, val2) TF_CHECK_OP(Check_NE, !=, val1, val2)
#define TF_CHECK_LE(val1, val2) TF_CHECK_OP(Check_LE, <=, val1, val2)
#define TF_CHECK_LT(val1, val2) TF_CHECK_OP(Check_LT, <, val1, val2)
#define TF_CHECK_GE(val1, val2) TF_CHECK_OP(Check_GE, >=, val1, val2)
#define TF_CHECK_GT(val1, val2) TF_CHECK_OP(Check_GT, >, val1, val2)

#define TF_CHECK_NOTNULL(val) TF_CHECK(val != nullptr)

}  // namespace internal
}  // namespace xla

#endif  // XLA_CLIENT_TF_LOGGING_H_
