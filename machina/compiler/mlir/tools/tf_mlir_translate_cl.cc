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

#include "machina/compiler/mlir/tools/tf_mlir_translate_cl.h"

#include <string>

#include "toolchain/Support/CommandLine.h"

// These command-line options are following LLVM conventions because we also
// need to register the TF Graph(Def) to MLIR conversion with mlir-translate,
// which expects command-line options of such style.

using toolchain::cl::opt;

// Import options.
// NOLINTNEXTLINE
opt<std::string> input_arrays(
    "tf-input-arrays", toolchain::cl::desc("Input tensor names, separated by ','"),
    toolchain::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> input_dtypes(
    "tf-input-data-types",
    toolchain::cl::desc("(Optional) Input tensor data types, separated by ','. Use "
                   "'' if a single data type is skipped. The data type from "
                   "the import graph is used if it is skipped."),
    toolchain::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> input_shapes(
    "tf-input-shapes",
    toolchain::cl::desc(
        "Input tensor shapes. Shapes for different tensors are separated by "
        "':', and dimension sizes for the same tensor are separated by ','"),
    toolchain::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> output_arrays(
    "tf-output-arrays", toolchain::cl::desc("Output tensor names, separated by ','"),
    toolchain::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> control_output_arrays(
    "tf-control-output-arrays",
    toolchain::cl::desc("Control output node names, separated by ','"),
    toolchain::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> inference_type(
    "tf-inference-type",
    toolchain::cl::desc(
        "Sets the type of real-number arrays in the output file. Only allows "
        "float and quantized types"),
    toolchain::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> min_values(
    "tf-input-min-values",
    toolchain::cl::desc(
        "Sets the lower bound of the input data. Separated by ','; Each entry "
        "in the list should match an entry in -tf-input-arrays. This is "
        "used when -tf-inference-type is a quantized type."),
    toolchain::cl::Optional, toolchain::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> max_values(
    "tf-input-max-values",
    toolchain::cl::desc(
        "Sets the upper bound of the input data. Separated by ','; Each entry "
        "in the list should match an entry in -tf-input-arrays. This is "
        "used when -tf-inference-type is a quantized type."),
    toolchain::cl::Optional, toolchain::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> debug_info_file(
    "tf-debug-info",
    toolchain::cl::desc("Path to the debug info file of the input graph def"),
    toolchain::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> xla_compile_device_type(
    "tf-xla-compile-device-type",
    toolchain::cl::desc("Sets the compilation device type of the input graph def"),
    toolchain::cl::init(""));

// TODO(b/134792656): If pruning is moved into TF dialect as a pass
// we should remove this.
// NOLINTNEXTLINE
opt<bool> prune_unused_nodes(
    "tf-prune-unused-nodes",
    toolchain::cl::desc("Prune unused nodes in the input graphdef"),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
opt<bool> convert_legacy_fed_inputs(
    "tf-convert-legacy-fed-inputs",
    toolchain::cl::desc(
        "Eliminate LegacyFedInput nodes by replacing them with Placeholder"),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
opt<bool> graph_as_function("tf-graph-as-function",
                            toolchain::cl::desc("Treat main graph as a function"),
                            toolchain::cl::init(false));

// NOLINTNEXTLINE
opt<bool> upgrade_legacy("tf-upgrade-legacy",
                         toolchain::cl::desc("Upgrade legacy TF graph behavior"),
                         toolchain::cl::init(false));

// NOLINTNEXTLINE
opt<bool> enable_shape_inference(
    "tf-enable-shape-inference-on-import",
    toolchain::cl::desc("Enable shape inference on import (temporary)"),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
opt<bool> unconditionally_use_set_output_shapes(
    "tf-enable-unconditionally-use-set-output-shapes-on-import",
    toolchain::cl::desc("Enable using the _output_shapes unconditionally on import "
                   "(temporary)"),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
opt<bool> enable_soft_placement(
    "tf-enable-soft-placement-on-import",
    toolchain::cl::desc("Enable soft device placement on import."),
    toolchain::cl::init(false));

// NOLINTNEXTLINE
opt<bool> set_original_tf_func_name(
    "tf-set-original-tf-func-name-on-import",
    toolchain::cl::desc("Set original TF function name on importi."),
    toolchain::cl::init(false));

// Export options.
// NOLINTNEXTLINE
opt<bool> export_entry_func_to_flib(
    "tf-export-entry-func-to-flib",
    toolchain::cl::desc(
        "Export entry function to function library instead of graph"),
    toolchain::cl::init(false));
// NOLINTNEXTLINE
opt<bool> export_original_tf_func_name(
    "tf-export-original-func-name",
    toolchain::cl::desc("Export functions using the name set in the attribute "
                   "'tf._original_func_name' if it exists."),
    toolchain::cl::init(false));
