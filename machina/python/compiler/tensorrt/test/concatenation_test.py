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
"""Model script to test TF-TensorRT integration."""

import numpy as np

from machina.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.ops import array_ops
from machina.python.ops import gen_array_ops
from machina.python.ops import gen_math_ops
from machina.python.platform import test


class ConcatenationTest(trt_test.TfTrtIntegrationTestBase):
  """Testing Concatenation in TF-TRT conversion."""

  def GraphFn(self, x):
    dtype = x.dtype
    # scale
    a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtype)
    r1 = x / a
    a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtype)
    r2 = a / x
    a = constant_op.constant(np.random.randn(1, 3, 1), dtype=dtype)
    r3 = a + x
    a = constant_op.constant(np.random.randn(1, 3, 1), dtype=dtype)
    r4 = x * a
    a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtype)
    r5 = x - a
    a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtype)
    r6 = a - x
    a = constant_op.constant(np.random.randn(3, 1), dtype=dtype)
    r7 = x - a
    a = constant_op.constant(np.random.randn(3, 1), dtype=dtype)
    r8 = a - x
    a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtype)
    r9 = gen_math_ops.maximum(x, a)
    a = constant_op.constant(np.random.randn(3, 1), dtype=dtype)
    r10 = gen_math_ops.minimum(a, x)
    a = constant_op.constant(np.random.randn(3), dtype=dtype)
    r11 = x * a
    a = constant_op.constant(np.random.randn(1), dtype=dtype)
    r12 = a * x
    concat1 = array_ops.concat([r1, r2, r3, r4, r5, r6], axis=-1)
    concat2 = array_ops.concat([r7, r8, r9, r10, r11, r12], axis=3)
    x = array_ops.concat([concat1, concat2], axis=-1)
    return gen_array_ops.reshape(x, [2, -1], name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[2, 3, 3, 1]],
                            [[2, 126]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_000"]


if __name__ == "__main__":
  test.main()
