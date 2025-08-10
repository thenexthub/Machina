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
"""Tests for machina.kernels.bcast_ops."""

from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.ops.gen_array_ops import broadcast_args
from machina.python.ops.gen_array_ops import broadcast_gradient_args
from machina.python.platform import test


class BcastOpsTest(test.TestCase):

  def _GetBroadcastShape(self, xs, ys):
    return self.evaluate(broadcast_args(xs, ys))

  def _GetGradientArgs(self, xs, ys):
    return self.evaluate(broadcast_gradient_args(xs, ys))

  def testBasic(self):
    r = self._GetBroadcastShape([2, 3, 5], [1])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([1], [2, 3, 5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([2, 3, 5], [5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([5], [2, 3, 5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([2, 3, 5], [3, 5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([3, 5], [2, 3, 5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([2, 3, 5], [3, 1])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([3, 1], [2, 3, 5])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([2, 1, 5], [3, 1])
    self.assertAllEqual(r, [2, 3, 5])

    r = self._GetBroadcastShape([3, 1], [2, 1, 5])
    self.assertAllEqual(r, [2, 3, 5])

  def testBasicGradient(self):
    r0, r1 = self._GetGradientArgs([2, 3, 5], [1])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1, 2])

    r0, r1 = self._GetGradientArgs([1], [2, 3, 5])
    self.assertAllEqual(r0, [0, 1, 2])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 3, 5], [5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1])

    r0, r1 = self._GetGradientArgs([5], [2, 3, 5])
    self.assertAllEqual(r0, [0, 1])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 3, 5], [3, 5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0])

    r0, r1 = self._GetGradientArgs([3, 5], [2, 3, 5])
    self.assertAllEqual(r0, [0])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 3, 5], [3, 1])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 2])

    r0, r1 = self._GetGradientArgs([3, 1], [2, 3, 5])
    self.assertAllEqual(r0, [0, 2])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 1, 5], [3, 1])
    self.assertAllEqual(r0, [1])
    self.assertAllEqual(r1, [0, 2])

    r0, r1 = self._GetGradientArgs([3, 1], [2, 1, 5])
    self.assertAllEqual(r0, [0, 2])
    self.assertAllEqual(r1, [1])

  def testZeroDims(self):
    r = self._GetBroadcastShape([2, 0, 3, 0, 5], [3, 0, 5])
    self.assertAllEqual(r, [2, 0, 3, 0, 5])

    r = self._GetBroadcastShape([3, 0, 5], [2, 0, 3, 0, 5])
    self.assertAllEqual(r, [2, 0, 3, 0, 5])

    r = self._GetBroadcastShape([2, 0, 3, 0, 5], [3, 1, 5])
    self.assertAllEqual(r, [2, 0, 3, 0, 5])

    r = self._GetBroadcastShape([3, 1, 5], [2, 0, 3, 0, 5])
    self.assertAllEqual(r, [2, 0, 3, 0, 5])

  def testZeroDimsGradient(self):
    r0, r1 = self._GetGradientArgs([2, 0, 3, 0, 5], [3, 0, 5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1])

    r0, r1 = self._GetGradientArgs([3, 0, 5], [2, 0, 3, 0, 5])
    self.assertAllEqual(r0, [0, 1])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 0, 3, 0, 5], [3, 1, 5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1, 3])

    r0, r1 = self._GetGradientArgs([3, 1, 5], [2, 0, 3, 0, 5])
    self.assertAllEqual(r0, [0, 1, 3])
    self.assertAllEqual(r1, [])

  def testDataTypes(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      r = self._GetBroadcastShape(
          constant_op.constant([2, 3, 5], dtype=dtype),
          constant_op.constant([1], dtype=dtype))
      self.assertAllEqual(r, [2, 3, 5])

      r0, r1 = self._GetGradientArgs(
          constant_op.constant([2, 3, 5], dtype=dtype),
          constant_op.constant([1], dtype=dtype))
      self.assertAllEqual(r0, [])
      self.assertAllEqual(r1, [0, 1, 2])


if __name__ == "__main__":
  test.main()
