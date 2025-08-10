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

from machina.python.framework import ops
from machina.python.framework import test_util
from machina.python.ops import math_ops
from machina.python.ops import random_ops
from machina.python.ops import variables
from machina.python.platform import test
from machina.python.profiler.internal import flops_registry  # pylint: disable=unused-import


class FlopsRegistryTest(test.TestCase):

  @test_util.run_v1_only('Test requires a Graph and NodeDef inspection')
  def testSimpleStatistics(self):
    a = variables.Variable(random_ops.random_normal([25, 16]))
    b = variables.Variable(random_ops.random_normal([16, 9]))
    math_ops.matmul(a, b)
    g = ops.get_default_graph()
    for op in g.get_operations():
      flops = ops.get_stats_for_node_def(g, op.node_def, 'flops').value
      if op.name == 'MatMul':
        self.assertEqual(7200, flops)

  @test_util.run_v1_only('Test requires a Graph and NodeDef inspection')
  def testTransposedStatistics(self):
    a = variables.Variable(random_ops.random_normal([16, 25]))
    b = variables.Variable(random_ops.random_normal([16, 9]))
    math_ops.matmul(a, b, transpose_a=True)
    g = ops.get_default_graph()
    for op in g.get_operations():
      flops = ops.get_stats_for_node_def(g, op.node_def, 'flops').value
      if op.name == 'MatMul':
        self.assertEqual(7200, flops)


if __name__ == '__main__':
  test.main()
