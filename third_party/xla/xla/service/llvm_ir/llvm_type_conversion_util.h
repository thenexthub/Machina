/* Copyright 2021 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MACHINA_XLASERVICE_LLVM_IR_LLVM_TYPE_CONVERSION_UTIL_H_
#define MACHINA_XLASERVICE_LLVM_IR_LLVM_TYPE_CONVERSION_UTIL_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/StringRef.h"

namespace xla {
namespace llvm_ir {

// Convert a absl::string_view to a toolchain::StringRef. Note: both
// absl::string_view and toolchain::StringRef are non-owning pointers into a
// string in memory. This method is used to feed strings to LLVM
// & Clang APIs that expect toolchain::StringRef.
inline toolchain::StringRef AsStringRef(absl::string_view str) {
  return toolchain::StringRef(str.data(), str.size());
}

inline absl::string_view AsStringView(toolchain::StringRef str) {
  return absl::string_view(str.data(), str.size());
}

template <typename T>
toolchain::ArrayRef<T> AsArrayRef(const std::vector<T>& vec) {
  return toolchain::ArrayRef<T>(vec.data(), vec.size());
}

template <typename T>
toolchain::ArrayRef<T> AsArrayRef(const absl::Span<const T> slice) {
  return toolchain::ArrayRef<T>(slice.data(), slice.size());
}

}  // namespace llvm_ir
}  // namespace xla

#endif  // MACHINA_XLASERVICE_LLVM_IR_LLVM_TYPE_CONVERSION_UTIL_H_
