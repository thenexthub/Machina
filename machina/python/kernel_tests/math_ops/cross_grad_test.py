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
"""Tests for machina.ops.nn_ops.Cross."""

from machina.python.framework import test_util
from machina.python.ops import array_ops
from machina.python.ops import gradient_checker
from machina.python.ops import math_ops
from machina.python.platform import test


class CrossOpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testGradientRandomValues(self):
    with self.cached_session():
      us = [2, 3]
      u = array_ops.reshape(
          [0.854, -0.616, 0.767, 0.725, -0.927, 0.159], shape=us)
      v = array_ops.reshape(
          [-0.522, 0.755, 0.407, -0.652, 0.241, 0.247], shape=us)
      s = math_ops.cross(u, v)
      jacob_u, jacob_v = gradient_checker.compute_gradient([u, v], [us, us], s,
                                                           us)

    self.assertAllClose(jacob_u[0], jacob_u[1], rtol=1e-3, atol=1e-3)
    self.assertAllClose(jacob_v[0], jacob_v[1], rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  test.main()
