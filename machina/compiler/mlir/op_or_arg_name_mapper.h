/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_OP_OR_ARG_NAME_MAPPER_H_
#define MACHINA_COMPILER_MLIR_OP_OR_ARG_NAME_MAPPER_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/PointerUnion.h"
#include "toolchain/ADT/StringMap.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain

namespace machina {

// PointerUnion for operation and value.
// TODO(jpienaar): Rename the files.
using OpOrVal = toolchain::PointerUnion<mlir::Operation*, mlir::Value>;

// Mapper from operation or value to name.
class OpOrArgNameMapper {
 public:
  // Returns mapped name for the operation or value.
  std::optional<toolchain::StringRef> GetMappedName(OpOrVal op_or_val);

  // Returns mapped name for the operation or value as a string_view.
  std::optional<absl::string_view> GetMappedNameView(OpOrVal op_or_val);

  // Returns unique name for the given prefix.
  toolchain::StringRef GetUniqueName(toolchain::StringRef prefix, int hash_value = 0);

  // Returns unique name for the operation or value.
  toolchain::StringRef GetUniqueName(OpOrVal op_or_val, int hash_value = 0);

  // Returns unique name as a string_view for the operation or value.
  absl::string_view GetUniqueNameView(OpOrVal op_or_val);

  // Initializes operation or value to map to name. Returns number of
  // operations or value already named 'name' which should be 0 else
  // GetUniqueName could return the same names for different operations or
  // values.
  // Note: Its up to the caller to decide the behavior when assigning two
  // operations or values to the same name.
  int InitOpName(OpOrVal op_or_val, toolchain::StringRef name);

  virtual ~OpOrArgNameMapper();

 protected:
  // Returns true if the name is unique. A derived class can override it if the
  // class maintains uniqueness in a different scope.
  virtual bool IsUnique(toolchain::StringRef name);

  // Returns a constant view of the underlying map.
  const toolchain::DenseMap<OpOrVal, absl::string_view>& GetMap() const {
    return op_or_val_to_name_;
  }

  // Returns the separator used before uniqueing suffix.
  virtual toolchain::StringRef GetSuffixSeparator() { return ""; }

  virtual toolchain::StringRef GetDashSeparator() { return "_"; }

 private:
  // Returns name from the location of the operation or value.
  virtual std::string GetName(OpOrVal op_or_val) = 0;

  // Maps string name to count. This map is used to help keep track of unique
  // names for operations or values.
  toolchain::StringMap<int64_t> name_to_count_;
  // Maps operation or values to name. Value in map is a view of the string
  // name in `name_to_count_`. Names in `name_to_count_` are never removed.
  toolchain::DenseMap<OpOrVal, absl::string_view> op_or_val_to_name_;
};

// OpOrArgNameMapper that returns, for operations or values not initialized
// to a specific name, a name based on the location of the operation or
// value.
class OpOrArgLocNameMapper : public OpOrArgNameMapper {
 protected:
  std::string GetName(OpOrVal op_or_val) override;
};

// OpOrArgNameMapper that returns, for operations or values not initialized
// to a specific name, a short name.
class OpOrArgStripNameMapper : public OpOrArgNameMapper {
 private:
  std::string GetName(OpOrVal op_or_val) override;

  // Number of ops mapped.
  int count_ = 0;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_OP_OR_ARG_NAME_MAPPER_H_
