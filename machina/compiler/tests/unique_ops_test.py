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
"""Tests for unique ops."""

import numpy as np

from machina.compiler.tests import xla_test
from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.ops import array_ops
from machina.python.ops import gen_array_ops
from machina.python.platform import googletest


class UniqueTest(xla_test.XLATestCase):

  def testNegativeAxis(self):
    """Verifies that an axis with negative index is converted to positive."""
    with self.session() as session:
      with self.test_scope():
        px = array_ops.placeholder(dtypes.float32, [2, 1, 1], name="x")
        axis = constant_op.constant([-1], dtype=dtypes.int32)
        output = gen_array_ops.unique_v2(px, axis)
      result = session.run(
          output, {px: np.array([[[-2.0]], [[10.0]]], dtype=np.float32)}
      )
      self.assertAllEqual(
          result.y, np.array([[[-2.0]], [[10.0]]], dtype=np.float32)
      )
      self.assertAllEqual(result.idx, np.array([0], dtype=np.int32))


if __name__ == "__main__":
  googletest.main()
