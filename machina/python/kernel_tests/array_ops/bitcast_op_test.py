###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Tuesday, March 25, 2025.                                            #
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
"""Tests for tf.bitcast."""

import sys

import numpy as np

from machina.python.framework import dtypes
from machina.python.framework import errors
from machina.python.framework import ops
from machina.python.framework import test_util
from machina.python.ops import array_ops
from machina.python.platform import test


@test_util.with_eager_op_as_function
class BitcastTest(test.TestCase):

  def _testBitcast(self, x, datatype, shape):
    with test_util.use_gpu():
      tf_ans = array_ops.bitcast(x, datatype)
      out = self.evaluate(tf_ans)
      if sys.byteorder == "little":
        buff_after = memoryview(out).tobytes()
        buff_before = memoryview(x).tobytes()
      else:
        buff_after = memoryview(out.byteswap()).tobytes()
        buff_before = memoryview(x.byteswap()).tobytes()
      self.assertEqual(buff_before, buff_after)
      self.assertEqual(tf_ans.get_shape(), shape)
      self.assertEqual(tf_ans.dtype, datatype)

  def testSmaller(self):
    x = np.random.rand(3, 2)
    datatype = dtypes.int8
    shape = [3, 2, 8]
    self._testBitcast(x, datatype, shape)

  def testLarger(self):
    x = np.arange(16, dtype=np.int8).reshape([4, 4])
    datatype = dtypes.int32
    shape = [4]
    self._testBitcast(x, datatype, shape)

  def testSameDtype(self):
    x = np.random.rand(3, 4)
    shape = [3, 4]
    self._testBitcast(x, x.dtype, shape)

  def testSameSize(self):
    x = np.random.rand(3, 4)
    shape = [3, 4]
    self._testBitcast(x, dtypes.int64, shape)

  def testErrors(self):
    x = np.zeros([1, 1], np.int8)
    datatype = dtypes.int32
    # When eager_op_as_function is enabled shape inference will raise
    # a different more informative error message.
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "Cannot bitcast from 6 to 3|convert from s8.* to S32"):
      array_ops.bitcast(x, datatype, None)

  def testEmpty(self):
    x = np.ones([], np.int32)
    datatype = dtypes.int8
    shape = [4]
    self._testBitcast(x, datatype, shape)

  def testUnknownShape(self):
    # Need to use placeholder for unknown shape
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32)
      datatype = dtypes.int8
      array_ops.bitcast(x, datatype, None)

  @test_util.disable_tfrt("b/169901260")
  def testQuantizedType(self):
    shape = [3, 4]
    x = np.zeros(shape, np.uint16)
    datatype = dtypes.quint16
    self._testBitcast(x, datatype, shape)

  def testUnsignedType(self):
    shape = [3, 4]
    x = np.zeros(shape, np.int64)
    datatype = dtypes.uint64
    self._testBitcast(x, datatype, shape)


if __name__ == "__main__":
  test.main()
