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

#ifndef MACHINA_PYTHON_FRAMEWORK_PYTHON_OP_GEN_ANNOTATOR_H_
#define MACHINA_PYTHON_FRAMEWORK_PYTHON_OP_GEN_ANNOTATOR_H_

#include <unordered_map>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/platform/types.h"
#include "machina/python/framework/op_reg_offset.pb.h"

namespace machina {
namespace python_op_gen_internal {

inline constexpr absl::string_view kKytheCorpus = "github.com/machina/machina";

// Collects and builds the generated code metadata for Kythe indexing.
class GeneratedCodeAnnotator {
 public:
  // Adds annotation of generated function and calculates the offset of
  // generated source based on base_pos_.
  void AddAnnotation(const OpDef& op_def, absl::string_view function_name,
                     uint32_t offset_start);
  // Updates base cursor.
  void SetBase(uint32_t pos) { base_pos_ = pos; }
  // Builds Kythe metadata from the offset map.
  string BuildKytheMetadata();
  // Fills the source offsets from OpRegOffsets.
  void FillSourceOffsets(const OpRegOffsets& op_reg_offsets);

  // Structure to store byte offsets of generated symbols.
  struct ByteOffsets {
    // The offsets of the symbol in the source file. Only valid if file_path
    // is set.
    uint32_t source_start = 0;
    uint32_t source_end = 0;
    // The offsets of the symbol in the generated file.
    uint32_t generated_start = 0;
    uint32_t generated_end = 0;
    // The file path of the source file.
    string file_path;
  };

 private:
  uint32_t base_pos_ = 0;
  std::unordered_map<string, ByteOffsets> byte_offsets_map_;
};

}  // namespace python_op_gen_internal
}  // namespace machina

#endif  // MACHINA_PYTHON_FRAMEWORK_PYTHON_OP_GEN_ANNOTATOR_H_
