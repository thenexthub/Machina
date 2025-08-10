###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Friday, April 11, 2025.                                             #
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
"""Tests for function_wrappers module."""

from machina.python.autograph.core import converter
from machina.python.autograph.core import function_wrappers
from machina.python.eager import context
from machina.python.framework import constant_op
from machina.python.ops import variables
from machina.python.platform import test


class FunctionWrappersTest(test.TestCase):

  def test_name_scope(self):
    if context.executing_eagerly():
      self.skipTest('Tensor names are disabled in eager')

    with function_wrappers.FunctionScope(
        'test_name', None,
        converter.ConversionOptions(
            optional_features=converter.Feature.NAME_SCOPES)):
      t = constant_op.constant(1)
    self.assertIn('test_name', t.name)

  def test_auto_control_deps(self):
    v = variables.Variable(1)
    with function_wrappers.FunctionScope(
        '_', None,
        converter.ConversionOptions(
            optional_features=converter.Feature.AUTO_CONTROL_DEPS)) as scope:
      v.assign(2)
      op = scope.ret(constant_op.constant(1), True)
    self.evaluate(op)
    self.assertEqual(self.evaluate(v.read_value()), 2)

  def test_all_disabled(self):
    with function_wrappers.FunctionScope(None, None,
                                         converter.STANDARD_OPTIONS):
      t = constant_op.constant(1)
    self.assertEqual(self.evaluate(t), 1)


if __name__ == '__main__':
  test.main()
