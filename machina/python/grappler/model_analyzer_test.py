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
"""Tests for the cost analyzer."""

from machina.python.framework import constant_op
from machina.python.framework import meta_graph
from machina.python.framework import ops
from machina.python.framework import test_util
from machina.python.grappler import model_analyzer
from machina.python.ops import math_ops
from machina.python.platform import test


class PyWrapOptimizeGraphTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testBasic(self):
    """Make sure arguments can be passed correctly."""
    a = constant_op.constant([10, 11], name="a")
    b = constant_op.constant([10], name="b")
    c = math_ops.add(a, b, name="c")
    d = math_ops.add_n([a, c], name="d")
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(d)
    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())

    report = model_analyzer.GenerateModelReport(mg)

    # Check the report headers
    self.assertIn(b"a [Const]", report)
    self.assertIn(b"a [Const]", report)
    self.assertIn(b"c [AddV2]", report)
    self.assertIn(b"d [AddN]", report)

    # Also print the report to make it easier to debug
    print("{}".format(report))

  @test_util.run_deprecated_v1
  def testDebugMode(self):
    """Make sure arguments can be passed correctly."""
    a = constant_op.constant([10, 11], name="a")
    b = constant_op.constant([10], name="b")
    c = math_ops.add(a, b, name="c")
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(c)
    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())

    report = model_analyzer.GenerateModelReport(mg, debug=True)

    # Check the report headers
    self.assertIn(b"input 0 (int32) has known value", report)
    self.assertIn(b"input 1 (int32) has known value", report)

    # Also print the report to make it easier to debug
    print("{}".format(report))


if __name__ == "__main__":
  test.main()
