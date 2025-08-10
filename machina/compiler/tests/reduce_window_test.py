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
"""Tests for xla.reduce_window."""

import numpy as np

from machina.compiler.tests import xla_test
from machina.compiler.tf2xla.python import xla
from machina.python.framework import dtypes
from machina.python.framework import function
from machina.python.ops import array_ops
from machina.python.platform import googletest


class ReduceWindowTest(xla_test.XLATestCase):
  """Test cases for xla.reduce_window."""

  def _reduce_window(self, operand, init, reducer, **kwargs):
    with self.session():
      placeholder = array_ops.placeholder(operand.dtype)
      with self.test_scope():
        output = xla.reduce_window(placeholder, init, reducer, **kwargs)
      return output.eval(feed_dict={placeholder: operand})

  def testReduceWindow(self):

    # TODO(b/77644762): float16 and float64 ReduceWindow are unimplemented.
    for dtype in set(self.numeric_types).intersection(
        set([dtypes.bfloat16.as_numpy_dtype, np.float32])):

      @function.Defun(dtype, dtype)
      def sum_reducer(x, y):
        return x + y

      @function.Defun(dtype, dtype)
      def mul_reducer(x, y):
        return x * y

      self.assertAllClose(
          np.array([3, 5, 7, 9, 11, 13], dtype=dtype),
          self._reduce_window(
              np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype),
              0.0,
              sum_reducer,
              window_dimensions=[2]))

      self.assertAllClose(
          np.array([3, 7, 11], dtype=dtype),
          self._reduce_window(
              np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype),
              0.0,
              sum_reducer,
              window_dimensions=[2],
              window_strides=[2]))

      self.assertAllClose(
          np.array([1, 4, 7], dtype=dtype),
          self._reduce_window(
              np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype),
              0.0,
              sum_reducer,
              window_dimensions=[1],
              window_strides=[3]))

      self.assertAllClose(
          np.array([[24, 36, 24], [96, 0, 0]], dtype=dtype),
          self._reduce_window(
              np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 0, 1]], dtype=dtype),
              1.0,
              mul_reducer,
              window_dimensions=[2, 2],
              window_strides=[1, 1]))

      self.assertAllClose(
          np.array([[0, 0, 0], [5, 10, 5], [2, 4, 1], [0, 0, 0]], dtype=dtype),
          self._reduce_window(
              np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 0, 1]], dtype=dtype),
              0.0,
              sum_reducer,
              window_dimensions=[2, 2],
              window_strides=[2, 2],
              padding=[[2, 3], [1, 2]]))


if __name__ == '__main__':
  googletest.main()
