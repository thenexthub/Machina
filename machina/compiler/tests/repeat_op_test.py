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
"""Tests for machina.ops.array_ops.repeat."""

from machina.compiler.tests import xla_test
from machina.python.eager import def_function
from machina.python.framework import dtypes
from machina.python.ops import array_ops
from machina.python.platform import test


class RepeatTest(xla_test.XLATestCase):

  def test(self):

    # Verifies that bounded dynamic result generated from the Where op can be
    # Reshaped correctly.
    @def_function.function(jit_compile=True)
    def repeat(values, repeats, axis):
      return array_ops.repeat(values, repeats, axis)

    with self.session() as sess:
      with self.test_scope():
        values = array_ops.constant([[1, 2], [3, 4]], dtype=dtypes.int32)
        repeats = array_ops.constant([1, 2], dtype=dtypes.int32)
        y1 = repeat(values, repeats, 0)
        y2 = repeat(values, repeats, 1)
      actual1, actual2 = sess.run([y1, y2])

    self.assertAllEqual(actual1, [[1, 2], [3, 4], [3, 4]])
    self.assertAllEqual(actual2, [[1, 2, 2], [3, 4, 4]])


if __name__ == "__main__":
  test.main()
