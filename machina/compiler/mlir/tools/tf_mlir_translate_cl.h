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

#ifndef MACHINA_COMPILER_MLIR_TOOLS_TF_MLIR_TRANSLATE_CL_H_
#define MACHINA_COMPILER_MLIR_TOOLS_TF_MLIR_TRANSLATE_CL_H_

// This file contains command-line options aimed to provide the parameters
// required by the TensorFlow Graph(Def) to MLIR module conversion. It is only
// intended to be included by binaries.

#include <string>

#include "toolchain/Support/CommandLine.h"

// Please see the implementation file for documentation of these options.

// Import options.
extern toolchain::cl::opt<std::string> input_arrays;
extern toolchain::cl::opt<std::string> input_dtypes;
extern toolchain::cl::opt<std::string> input_shapes;
extern toolchain::cl::opt<std::string> output_arrays;
extern toolchain::cl::opt<std::string> control_output_arrays;
extern toolchain::cl::opt<std::string> inference_type;
extern toolchain::cl::opt<std::string> min_values;
extern toolchain::cl::opt<std::string> max_values;
extern toolchain::cl::opt<std::string> debug_info_file;
extern toolchain::cl::opt<std::string> xla_compile_device_type;
extern toolchain::cl::opt<bool> prune_unused_nodes;
extern toolchain::cl::opt<bool> convert_legacy_fed_inputs;
extern toolchain::cl::opt<bool> graph_as_function;
extern toolchain::cl::opt<bool> upgrade_legacy;
// TODO(jpienaar): Temporary flag, flip default and remove.
extern toolchain::cl::opt<bool> enable_shape_inference;
extern toolchain::cl::opt<bool> unconditionally_use_set_output_shapes;
extern toolchain::cl::opt<bool> enable_soft_placement;
extern toolchain::cl::opt<bool> set_original_tf_func_name;

// Export options.
extern toolchain::cl::opt<bool> export_entry_func_to_flib;
extern toolchain::cl::opt<bool> export_original_tf_func_name;

#endif  // MACHINA_COMPILER_MLIR_TOOLS_TF_MLIR_TRANSLATE_CL_H_
