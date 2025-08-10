###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, July 14, 2025.                                              #
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
"""Functions used by multiple tflite test files."""

from machina.lite.python import schema_py_generated as schema_fb
from machina.lite.python import schema_util
from machina.lite.tools import visualize


def get_ops_list(model_data):
  """Returns a set of ops in the tflite model data."""
  model = schema_fb.Model.GetRootAsModel(model_data, 0)
  op_set = set()

  for subgraph_idx in range(model.SubgraphsLength()):
    subgraph = model.Subgraphs(subgraph_idx)
    for op_idx in range(subgraph.OperatorsLength()):
      op = subgraph.Operators(op_idx)
      opcode = model.OperatorCodes(op.OpcodeIndex())
      builtin_code = schema_util.get_builtin_code_from_operator_code(opcode)
      if builtin_code == schema_fb.BuiltinOperator.CUSTOM:
        opname = opcode.CustomCode().decode("utf-8")
        op_set.add(opname)
      else:
        op_set.add(visualize.BuiltinCodeToName(builtin_code))
  return op_set


def get_output_shapes(model_data):
  """Returns a list of output shapes in the tflite model data."""
  model = schema_fb.Model.GetRootAsModel(model_data, 0)

  output_shapes = []
  for subgraph_idx in range(model.SubgraphsLength()):
    subgraph = model.Subgraphs(subgraph_idx)
    for output_idx in range(subgraph.OutputsLength()):
      output_tensor_idx = subgraph.Outputs(output_idx)
      output_tensor = subgraph.Tensors(output_tensor_idx)
      output_shapes.append(output_tensor.ShapeAsNumpy().tolist())

  return output_shapes
