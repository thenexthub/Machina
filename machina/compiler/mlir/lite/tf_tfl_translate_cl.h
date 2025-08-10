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

#ifndef MACHINA_COMPILER_MLIR_LITE_TF_TFL_TRANSLATE_CL_H_
#define MACHINA_COMPILER_MLIR_LITE_TF_TFL_TRANSLATE_CL_H_

// This file contains command-line options aimed to provide the parameters
// required by the TensorFlow Graph(Def) to TF Lite Flatbuffer conversion. It is
// only intended to be included by binaries.

#include <string>

#include "toolchain/Support/CommandLine.h"

// The commandline options are defined in LLVM style, so the caller should
// use toolchain::InitLLVM to initialize the options.
//
// Please see the implementation file for documentation of details of these
// options.
// TODO(jpienaar): Revise the command line option parsing here.
extern toolchain::cl::opt<std::string> input_file_name;
extern toolchain::cl::opt<std::string> output_file_name;
extern toolchain::cl::opt<bool> use_splatted_constant;
extern toolchain::cl::opt<bool> input_mlir;
extern toolchain::cl::opt<bool> output_mlir;
extern toolchain::cl::list<std::string> custom_opdefs;
extern toolchain::cl::opt<bool> emit_quant_adaptor_ops;
extern toolchain::cl::opt<std::string> quant_stats_file_name;
extern toolchain::cl::opt<bool> convert_tf_while_to_tfl_while;
extern toolchain::cl::opt<std::string> select_user_tf_ops;
extern toolchain::cl::opt<bool> allow_all_select_tf_ops;
extern toolchain::cl::opt<bool> unfold_batchmatmul;
extern toolchain::cl::opt<bool> unfold_large_splat_constant;
extern toolchain::cl::opt<bool> guarantee_all_funcs_one_use;
extern toolchain::cl::opt<bool> enable_dynamic_update_slice;
extern toolchain::cl::opt<bool> preserve_assert_op;
extern toolchain::cl::opt<bool> legalize_custom_tensor_list_ops;
extern toolchain::cl::opt<bool> reduce_type_precision;
extern toolchain::cl::opt<std::string> input_arrays;
extern toolchain::cl::opt<std::string> input_dtypes;
extern toolchain::cl::opt<std::string> input_shapes;
extern toolchain::cl::opt<std::string> output_arrays;
extern toolchain::cl::opt<std::string> control_output_arrays;
extern toolchain::cl::opt<std::string> inference_type;
extern toolchain::cl::opt<std::string> min_values;
extern toolchain::cl::opt<std::string> max_values;
extern toolchain::cl::opt<std::string> debug_info_file;
extern toolchain::cl::opt<bool> upgrade_legacy;
extern toolchain::cl::opt<bool> enable_shape_inference;

// Import saved model.
extern toolchain::cl::opt<bool> import_saved_model_object_graph;
extern toolchain::cl::opt<bool> import_saved_model_signature_defs;
extern toolchain::cl::opt<std::string> saved_model_tags;
extern toolchain::cl::opt<std::string> saved_model_exported_names;

// Import HLO.
enum HloImportType { proto, hlotxt, mlir_text };

extern toolchain::cl::opt<bool> import_hlo;
extern toolchain::cl::opt<HloImportType> hlo_import_type;

// enable_hlo_to_tf_conversion and disable_hlo_to_tfl_conversion are used to
// control the HLO to TF and HLO to TFLite conversion while debugging an
// input_mlir. The default value of enable_hlo_to_tf_conversion is false, and
// the default value of disable_hlo_to_tfl_conversion is true.
extern toolchain::cl::opt<bool> enable_hlo_to_tf_conversion;
extern toolchain::cl::opt<bool> disable_hlo_to_tfl_conversion;

// quantization related flags
extern toolchain::cl::opt<bool> post_training_quantization;

// TF to stablehlo pass flags
extern toolchain::cl::opt<bool> enable_stablehlo_conversion;

// Whether to enable the attempt to directly lower composites into tflite ops or
// not.
extern toolchain::cl::opt<bool> enable_composite_direct_lowering;

// The source model type
extern toolchain::cl::opt<std::string> model_origin_framework;

#endif  // MACHINA_COMPILER_MLIR_LITE_TF_TFL_TRANSLATE_CL_H_
