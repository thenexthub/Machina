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
"""Tests for Grappler Arithmetic Optimizer."""

from machina.python.eager import context
from machina.python.eager import def_function
from machina.python.ops import array_ops
from machina.python.ops import math_ops
from machina.python.platform import test


class ArithmeticOptimizerTest(test.TestCase):

  # See b/146524878.
  def testFunctionArgShapeInference(self):

    @def_function.function
    def f(x, y):
      return math_ops.matmul(
          x, array_ops.reshape(array_ops.transpose(y), [384, 1536]))

    with context.eager_mode():
      x = array_ops.ones((1, 384))
      y = array_ops.ones((1536, 384))
      with context.collect_graphs(optimized=True) as graphs:
        f(x, y).numpy()
      self.assertLen(graphs, 1)
      self.assertLen(graphs[0].node, 4)
      self.assertEqual(graphs[0].node[2].name,
                       'ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul')


if __name__ == '__main__':
  test.main()
