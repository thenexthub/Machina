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

#ifndef MACHINA_COMPILER_AOT_FLAGS_H_
#define MACHINA_COMPILER_AOT_FLAGS_H_

#include <string>
#include <vector>

#include "machina/core/util/command_line_flags.h"

namespace machina {
namespace tfcompile {

// Flags for the tfcompile binary.  See *.cc file for descriptions.

struct MainFlags {
  string graph;
  string debug_info;
  string debug_info_path_begin_marker;
  string config;
  bool dump_fetch_nodes = false;
  string target_triple;
  string target_cpu;
  string target_features;
  string entry_point;
  string cpp_class;
  string out_function_object;
  string out_metadata_object;
  string out_header;
  string out_constant_buffers_object;
  string out_session_module;
  string mlir_components;
  bool experimental_quantize = false;

  // Sanitizer pass options
  bool sanitize_dataflow = false;
  string sanitize_abilists_dataflow;

  // C++ codegen options
  bool gen_name_to_index = false;
  bool gen_program_shape = false;
  bool use_xla_nanort_runtime = false;
};

// Appends to flag_list a machina::Flag for each field in MainFlags.
void AppendMainFlags(std::vector<Flag>* flag_list, MainFlags* flags);

}  // namespace tfcompile
}  // namespace machina

#endif  // MACHINA_COMPILER_AOT_FLAGS_H_
