###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Tuesday, March 25, 2025.                                            #
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
"""Quantize training for TensorFlow."""
from machina.core.framework import graph_pb2
from machina.python._pywrap_quantize_training import DoQuantizeTrainingOnGraphDefHelper
from machina.python.util import deprecation
from machina.python.util.tf_export import tf_export


# Migrated this python code from deprecated quantize_training.i
@deprecation.deprecated(
    None,
    "GraphDef quantized training rewriter is deprecated in the long term.")
@tf_export(v1=["train.do_quantize_training_on_graphdef"])
def do_quantize_training_on_graphdef(input_graph, num_bits):
  """A general quantization scheme is being developed in `tf.contrib.quantize`.

  Consider using that instead, though since it is in the tf.contrib namespace,
  it is not subject to backward compatibility guarantees.

  Args:
    input_graph: A `GraphDef`.
    num_bits: The number of bits for quantize training.

  Returns:
    The graph with quantize training done.
  """

  graph = graph_pb2.GraphDef()
  result_graph_string = DoQuantizeTrainingOnGraphDefHelper(
      input_graph.SerializeToString(), num_bits)

  graph.ParseFromString(result_graph_string)
  return graph
