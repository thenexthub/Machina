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
"""Tests for autograph_ops module."""

from machina.python.framework import constant_op
from machina.python.ops import autograph_ops
from machina.python.platform import test


class AutographOpsTest(test.TestCase):

  def test_wrap_py_func_dummy_return(self):
    side_counter = [0]

    def test_fn(_):
      side_counter[0] += 1

    with self.cached_session():
      result = autograph_ops.wrap_py_func(test_fn, (5,))
      self.assertEqual(1, self.evaluate(result))
      self.assertEqual([1], side_counter)
      result = autograph_ops.wrap_py_func(test_fn, (constant_op.constant(5),))
      self.assertEqual(1, self.evaluate(result))
      self.assertEqual([2], side_counter)


if __name__ == '__main__':
  test.main()
