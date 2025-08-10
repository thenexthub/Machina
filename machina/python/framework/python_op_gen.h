/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_PYTHON_FRAMEWORK_PYTHON_OP_GEN_H_
#define MACHINA_PYTHON_FRAMEWORK_PYTHON_OP_GEN_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/types/span.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/framework/op_gen_lib.h"
#include "machina/core/platform/types.h"
#include "machina/python/framework/op_reg_offset.pb.h"

namespace machina {

// Returns a string containing the generated Python code for the given Ops.
// ops is a protobuff, typically generated using OpRegistry::Global()->Export.
// api_defs is typically constructed directly from ops.
// hidden_ops should be a list of Op names that should get a leading _
// in the output.
// source_file_list is optional and contains the name of the original C++ source
// file where the ops' REGISTER_OP() calls reside.
// op_reg_offsets contains the location of the ops' REGISTER_OP() calls
// in the file. If specified, returned string will contain a metadata comment
// which contains indexing information for Kythe.
string GetPythonOps(const OpList& ops, const ApiDefMap& api_defs,
                    const OpRegOffsets& op_reg_offsets,
                    absl::Span<const string> hidden_ops,
                    absl::Span<const string> source_file_list);

// Prints the output of GetPrintOps to stdout.
// hidden_ops should be a list of Op names that should get a leading _
// in the output.
// Optional fourth argument is the name of the original C++ source file
// where the ops' REGISTER_OP() calls reside.
void PrintPythonOps(const OpList& ops, const ApiDefMap& api_defs,
                    const OpRegOffsets& op_reg_offsets,
                    absl::Span<const string> hidden_ops,
                    absl::Span<const string> source_file_list);

// Get the python wrappers for a list of ops in a OpList.
// `op_list_buf` should be a pointer to a buffer containing
// the binary encoded OpList proto, and `op_list_len` should be the
// length of that buffer.
string GetPythonWrappers(const char* op_list_buf, size_t op_list_len);

// Get the type annotation for an arg
// `arg` should be an input or output of an op
// `type_annotations` should contain attr names mapped to TypeVar names
string GetArgAnnotation(
    const OpDef::ArgDef& arg,
    const std::unordered_map<string, string>& type_annotations);

}  // namespace machina

#endif  // MACHINA_PYTHON_FRAMEWORK_PYTHON_OP_GEN_H_
