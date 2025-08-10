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
"""Tests for Mean operators."""

from machina.python.eager.def_function import function
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.ops import array_ops
from machina.python.ops import math_ops
from machina.python.platform import test


class MeanOpTest(test.TestCase):
  def testReduceMeanOverflow(self):
    @function(jit_compile=True)
    def tpu_computation():
      shape = (4, 32, 512, 512, 96)
      dtype = dtypes.float32
      test_ones = array_ops.ones(shape=shape, dtype=dtype)
      return math_ops.reduce_mean(test_ones)
    with ops.device("TPU:0"):
      result = tpu_computation()
      self.assertAllClose(
          result, 1.0, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
  test.main()
