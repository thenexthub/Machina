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
from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.framework import tensor
from machina.python.ops import array_ops
from machina.python.ops import random_ops
from machina.python.ops.parallel_for import pfor
from machina.python.platform import test


class PForTest(test.TestCase):

  def test_rank_known(self):
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32, [None, None])
      rank = pfor._rank(x)
      self.assertIsInstance(rank, int)
      self.assertEqual(rank, 2)

  def test_rank_unknown(self):
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32)
      rank = pfor._rank(x)
      self.assertIsInstance(rank, tensor.Tensor)

  def test_size_known(self):
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32, [3, 5])
      size = pfor._size(x)
      self.assertIsInstance(size, int)
      self.assertEqual(size, 3 * 5)

  def test_size_unknown(self):
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32, [3, None])

      size = pfor._size(x, dtypes.int32)
      self.assertIsInstance(size, tensor.Tensor)
      self.assertEqual(size.dtype, dtypes.int32)

      size = pfor._size(x, dtypes.int64)
      self.assertIsInstance(size, tensor.Tensor)
      self.assertEqual(size.dtype, dtypes.int64)

  def test_expand_dims_static(self):
    x = random_ops.random_uniform([3, 5])
    axis = 1
    num_axes = 2
    expected = array_ops.reshape(x, [3, 1, 1, 5])
    actual = pfor._expand_dims(x, axis, num_axes)
    self.assertAllEqual(expected, actual)

  def test_expand_dims_dynamic(self):
    x = random_ops.random_uniform([3, 5])
    axis = 1
    num_axes = constant_op.constant([2])
    expected = array_ops.reshape(x, [3, 1, 1, 5])
    actual = pfor._expand_dims(x, axis, num_axes)
    self.assertAllEqual(expected, actual)


if __name__ == '__main__':
  test.main()
