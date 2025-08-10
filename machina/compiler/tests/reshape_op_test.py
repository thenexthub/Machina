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
"""Tests for reshape."""

from absl.testing import parameterized

from machina.compiler.tests import xla_test
from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.ops import array_ops
from machina.python.platform import googletest


class ReshapeTest(xla_test.XLATestCase, parameterized.TestCase):

  @parameterized.named_parameters(('32_bit_index', dtypes.int32),
                                  ('64_bit_index', dtypes.int64))
  def testBasic(self, index_dtype):
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[2, 3])
        with self.test_scope():
          shape = constant_op.constant([3, 2], dtype=index_dtype)
          o = array_ops.reshape(i, shape)
        params = {
            i: [[1, 2, 3], [4, 5, 6]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[1, 2], [3, 4], [5, 6]], result)

  def testInt64(self):
    with self.session():
      with self.test_scope():
        x = array_ops.zeros([50000, 50000], dtype=dtypes.bool)
        # Provide dimension larger than int32
        y = array_ops.reshape(x, [50000**2])
        self.assertEqual([50000**2], y.get_shape().as_list())
        # Even if first dimension is within int32, ensure we correctly go to
        # int64
        y = array_ops.reshape(x, [1, 50000**2])
        self.assertEqual([1, 50000**2], y.get_shape().as_list())


if __name__ == '__main__':
  googletest.main()
