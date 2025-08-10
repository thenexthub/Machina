/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file forward declares tfrt/support's template types. This
// file should not include any files from tfrt/support.

#ifndef TFRT_SUPPORT_FORWARD_DECLS_H_
#define TFRT_SUPPORT_FORWARD_DECLS_H_

#include <cstddef>
#include <memory>

#include "toolchain/Support/Casting.h"
#include "toolchain/Support/ErrorHandling.h"

// We don't forward declare:
//   DenseMap, SmallVector, StringMap, StringSet
// Because they use default template parameters.
namespace toolchain {

class raw_ostream;

template <typename T>
class ArrayRef;
template <typename T>
class Expected;
class Error;
template <typename T>
class MutableArrayRef;
template <class T>
using Optional = std::optional<T>;
class StringRef;

template <typename FunctionT>
class unique_function;
template <typename FunctionT>
class function_ref;
}  // namespace toolchain

namespace tsl {
class AsyncValue;
template <typename T>
class AsyncValueRef;
class Chain;
class ErrorAsyncValue;
class IndirectAsyncValue;
template <typename T>
class RCReference;
template <typename SubClass>
class ReferenceCounted;

namespace internal {
template <typename T>
class ConcurrentVector;
}  // namespace internal

}  // namespace tsl

namespace tfrt {

// Common TSL types.
using ::tsl::AsyncValue;                  // NOLINT
using ::tsl::AsyncValueRef;               // NOLINT
using ::tsl::Chain;                       // NOLINT
using ::tsl::ErrorAsyncValue;             // NOLINT
using ::tsl::IndirectAsyncValue;          // NOLINT
using ::tsl::RCReference;                 // NOLINT
using ::tsl::ReferenceCounted;            // NOLINT
using ::tsl::internal::ConcurrentVector;  // NOLINT

template <typename T>
using Expected = toolchain::Expected<T>;

using Error = toolchain::Error;

// Commonly used types imported from LLVM.
using raw_ostream = toolchain::raw_ostream;

template <typename T>
using ArrayRef = toolchain::ArrayRef<T>;
template <typename T>
using MutableArrayRef = toolchain::MutableArrayRef<T>;
template <class T>
using Optional = std::optional<T>;
using string_view = toolchain::StringRef;

// Casting operators.
using toolchain::cast;
using toolchain::cast_or_null;
using toolchain::dyn_cast;
using toolchain::dyn_cast_or_null;
using toolchain::isa;
using toolchain::isa_and_nonnull;

// TensorShape dimension type alias. It is currently fixed at 64 bits,
//  but might change to machine word in the future.
using Index = int64_t;

}  // namespace tfrt

#endif  // TFRT_SUPPORT_FORWARD_DECLS_H_
