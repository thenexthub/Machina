###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Saturday, May 31, 2025.                                             #
#                                                                             #
#   Licensed under the Apache License, Version 2.0 (the "License");           #
#   you may not use this file except in compliance with the License.          #
#   You may obtain a copy of the License at:                                  #
#                                                                             #
#       http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                             #
#   Unless required by applicable law or agreed to in writing, software       #
#   distributed under the License is distributed on an "AS IS" BASIS,         #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
#   See the License for the specific language governing permissions and       #
#   limitations under the License.                                            #
#                                                                             #
#   Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,            #
#   Middletown, DE 19709, New Castle County, USA.                             #
#                                                                             #
###############################################################################
"""Wraps TFLite Converter interface with python lazy loader."""
# We need to import pywrap_machina prior to the converter wrapper.
# pylint: disable=invalid-import-order,g-bad-import-order
from machina.python import pywrap_machina  # pylint: disable=unused-import
from machina.compiler.mlir.lite.python import _pywrap_converter_api
from machina.compiler.mlir.quantization.machina.python import py_function_lib


def wrapped_convert(
    model_flags_str,
    converter_flags_str,
    input_data_str,
    debug_info_str,
):
  """Wraps Convert with lazy loader."""
  return _pywrap_converter_api.Convert(
      model_flags_str,
      converter_flags_str,
      input_data_str,
      False,  # extended_return
      debug_info_str,
      py_function_lib.PyFunctionLibrary(),
  )


def wrapped_experimental_mlir_quantize(
    input_data_str,
    disable_per_channel,
    fully_quantize,
    inference_type,
    input_data_type,
    output_data_type,
    enable_numeric_verify,
    enable_whole_model_verify,
    denylisted_ops,
    denylisted_nodes,
    enable_variable_quantization,
    disable_per_channel_for_dense_layers,
    debug_options_str,
):
  """Wraps experimental mlir quantize model."""
  return _pywrap_converter_api.ExperimentalMlirQuantizeModel(
      input_data_str,
      disable_per_channel,
      fully_quantize,
      inference_type,
      input_data_type,
      output_data_type,
      enable_numeric_verify,
      enable_whole_model_verify,
      denylisted_ops,
      denylisted_nodes,
      enable_variable_quantization,
      disable_per_channel_for_dense_layers,
      debug_options_str,
  )


def wrapped_experimental_mlir_sparsify(input_data_str):
  """Wraps experimental mlir sparsify model."""
  return _pywrap_converter_api.ExperimentalMlirSparsifyModel(input_data_str)


def wrapped_register_custom_opdefs(custom_opdefs_list):
  """Wraps RegisterCustomOpdefs with lazy loader."""
  return _pywrap_converter_api.RegisterCustomOpdefs(custom_opdefs_list)


def wrapped_retrieve_collected_errors():
  """Wraps RetrieveCollectedErrors with lazy loader."""
  return _pywrap_converter_api.RetrieveCollectedErrors()


def wrapped_flat_buffer_file_to_mlir(model, input_is_filepath):
  """Wraps FlatBufferFileToMlir with lazy loader."""
  return _pywrap_converter_api.FlatBufferToMlir(model, input_is_filepath)
