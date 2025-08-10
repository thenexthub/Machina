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
"""Tests for the swig wrapper of items."""

from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.framework import errors_impl
from machina.python.framework import meta_graph
from machina.python.framework import ops
from machina.python.framework import tensor_shape
from machina.python.framework import test_util
from machina.python.grappler import item
from machina.python.ops import control_flow_ops
from machina.python.ops import gen_array_ops
from machina.python.ops import state_ops
from machina.python.ops import variable_v1
from machina.python.platform import test


class ItemTest(test.TestCase):

  def testInvalidItem(self):
    with ops.Graph().as_default() as g:
      a = constant_op.constant(10)
      b = constant_op.constant(20)
      c = a + b  # pylint: disable=unused-variable
      mg = meta_graph.create_meta_graph_def(graph=g)

    # The train op isn't specified: this should raise an InvalidArgumentError
    # exception.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      item.Item(mg)

  def testImportantOps(self):
    with ops.Graph().as_default() as g:
      a = constant_op.constant(10)
      b = constant_op.constant(20)
      c = a + b
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(c)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)
      op_list = grappler_item.IdentifyImportantOps()
      self.assertItemsEqual(['Const', 'Const_1', 'add'], op_list)

  def testOpProperties(self):
    with ops.Graph().as_default() as g:
      a = constant_op.constant(10)
      b = constant_op.constant(20)
      c = a + b
      z = control_flow_ops.no_op()
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(c)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)
      op_properties = grappler_item.GetOpProperties()

      # All the nodes in this model have one scalar output
      for node in grappler_item.metagraph.graph_def.node:
        node_prop = op_properties[node.name]

        if node.name == z.name:
          self.assertEqual(0, len(node_prop))
        else:
          self.assertEqual(1, len(node_prop))
          self.assertEqual(dtypes.int32, node_prop[0].dtype)
          self.assertEqual(tensor_shape.TensorShape([]), node_prop[0].shape)

  def testUpdates(self):
    with ops.Graph().as_default() as g:
      a = constant_op.constant(10)
      b = constant_op.constant(20)
      c = a + b
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(c)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)

    initial_tf_item = grappler_item.tf_item
    no_change_tf_item = grappler_item.tf_item
    self.assertEqual(initial_tf_item, no_change_tf_item)

    # Modify the placement.
    for node in grappler_item.metagraph.graph_def.node:
      node.device = '/cpu:0'
    new_tf_item = grappler_item.tf_item
    self.assertNotEqual(initial_tf_item, new_tf_item)

    # Assign the same placement.
    for node in grappler_item.metagraph.graph_def.node:
      node.device = '/cpu:0'
    newest_tf_item = grappler_item.tf_item
    self.assertEqual(new_tf_item, newest_tf_item)

  @test_util.run_v1_only('b/120545219')
  def testColocationConstraints(self):
    with ops.Graph().as_default() as g:
      c = constant_op.constant([10])
      v = variable_v1.VariableV1([3], dtype=dtypes.int32)
      i = gen_array_ops.ref_identity(v)
      a = state_ops.assign(i, c)
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(a)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)
      groups = grappler_item.GetColocationGroups()
      self.assertEqual(len(groups), 1)
      self.assertItemsEqual(
          groups[0], ['Assign', 'RefIdentity', 'Variable', 'Variable/Assign'])


if __name__ == '__main__':
  test.main()
