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
"""Tests for conditional_expressions module."""

from machina.python.autograph.operators import conditional_expressions
from machina.python.eager import def_function
from machina.python.framework import constant_op
from machina.python.framework import test_util
from machina.python.platform import test


def _basic_expr(cond):
  return conditional_expressions.if_exp(
      cond,
      lambda: constant_op.constant(1),
      lambda: constant_op.constant(2),
      'cond')


@test_util.run_all_in_graph_and_eager_modes
class IfExpTest(test.TestCase):

  def test_tensor(self):
    self.assertEqual(self.evaluate(_basic_expr(constant_op.constant(True))), 1)
    self.assertEqual(self.evaluate(_basic_expr(constant_op.constant(False))), 2)

  def test_tensor_mismatched_type(self):
    # tf.function required because eager cond degenerates to Python if.
    @def_function.function
    def test_fn():
      conditional_expressions.if_exp(
          constant_op.constant(True), lambda: 1.0, lambda: 2, 'expr_repr')

    with self.assertRaisesRegex(
        TypeError,
        "'expr_repr' has dtype float32 in the main.*int32 in the else"):
      test_fn()

  def test_python(self):
    self.assertEqual(self.evaluate(_basic_expr(True)), 1)
    self.assertEqual(self.evaluate(_basic_expr(False)), 2)
    self.assertEqual(
        conditional_expressions.if_exp(True, lambda: 1, lambda: 2, ''), 1)
    self.assertEqual(
        conditional_expressions.if_exp(False, lambda: 1, lambda: 2, ''), 2)


if __name__ == '__main__':
  test.main()
