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
"""Tests for exceptions module."""

from machina.python.autograph.operators import exceptions
from machina.python.framework import constant_op
from machina.python.framework import errors_impl
from machina.python.framework import test_util
from machina.python.platform import test


class ExceptionsTest(test.TestCase):

  def test_assert_tf_untriggered(self):
    with self.cached_session() as sess:
      t = exceptions.assert_stmt(
          constant_op.constant(True), lambda: constant_op.constant('ignored'))
      self.evaluate(t)

  @test_util.run_deprecated_v1
  def test_assert_tf_triggered(self):
    with self.cached_session() as sess:
      t = exceptions.assert_stmt(
          constant_op.constant(False),
          lambda: constant_op.constant('test message'))

      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  'test message'):
        self.evaluate(t)

  @test_util.run_deprecated_v1
  def test_assert_tf_multiple_printed_values(self):
    two_tensors = [
        constant_op.constant('test message'),
        constant_op.constant('another message')
    ]
    with self.cached_session() as sess:
      t = exceptions.assert_stmt(
          constant_op.constant(False), lambda: two_tensors)

      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  'test message.*another message'):
        self.evaluate(t)

  def test_assert_python_untriggered(self):
    side_effect_trace = []

    def expression_with_side_effects():
      side_effect_trace.append(object())
      return 'test message'

    exceptions.assert_stmt(True, expression_with_side_effects)

    self.assertListEqual(side_effect_trace, [])

  def test_assert_python_triggered(self):
    if not __debug__:
      # Python assertions only be tested when in debug mode.
      return

    side_effect_trace = []
    tracer = object()

    def expression_with_side_effects():
      side_effect_trace.append(tracer)
      return 'test message'

    with self.assertRaisesRegex(AssertionError, 'test message'):
      exceptions.assert_stmt(False, expression_with_side_effects)
    self.assertListEqual(side_effect_trace, [tracer])


if __name__ == '__main__':
  test.main()
