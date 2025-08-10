###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, May 19, 2025.                                               #
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
"""Test utilities for tf.signal."""

from machina.core.protobuf import config_pb2
from machina.python.grappler import tf_optimizer
from machina.python.training import saver


def grappler_optimize(graph, fetches=None, config_proto=None):
  """Tries to optimize the provided graph using grappler.

  Args:
    graph: A `tf.Graph` instance containing the graph to optimize.
    fetches: An optional list of `Tensor`s to fetch (i.e. not optimize away).
      Grappler uses the 'train_op' collection to look for fetches, so if not
      provided this collection should be non-empty.
    config_proto: An optional `tf.compat.v1.ConfigProto` to use when rewriting
      the graph.

  Returns:
    A `tf.compat.v1.GraphDef` containing the rewritten graph.
  """
  if config_proto is None:
    config_proto = config_pb2.ConfigProto()
    config_proto.graph_options.rewrite_options.min_graph_nodes = -1
  if fetches is not None:
    for fetch in fetches:
      graph.add_to_collection('train_op', fetch)
  metagraph = saver.export_meta_graph(graph_def=graph.as_graph_def())
  return tf_optimizer.OptimizeGraph(config_proto, metagraph)

