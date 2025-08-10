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
"""Test LRUCache by running different input batch sizes on same network."""

import numpy as np

from machina.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.framework import tensor_spec
from machina.python.ops import array_ops
from machina.python.ops import math_ops
from machina.python.ops import nn
from machina.python.platform import test


class LRUCacheTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, x):
    bias = constant_op.constant(
        np.random.randn(1, 10, 10, 1), dtype=dtypes.float32)
    x = math_ops.add(x, bias)
    x = nn.relu(x)
    return array_ops.identity(x, name="output")

  def GetParams(self):
    dtype = dtypes.float32
    input_dims = [[[1, 10, 10, 2]], [[2, 10, 10, 2]], [[4, 10, 10, 2]],
                  [[2, 10, 10, 2]]]
    expected_output_dims = [[[1, 10, 10, 2]], [[2, 10, 10, 2]], [[4, 10, 10,
                                                                  2]],
                            [[2, 10, 10, 2]]]
    return trt_test.TfTrtIntegrationTestParams(
        graph_fn=self.GraphFn,
        input_specs=[
            tensor_spec.TensorSpec([None, 10, 10, 2], dtypes.float32, "input")
        ],
        output_specs=[
            tensor_spec.TensorSpec([None, 10, 10, 1], dtypes.float32, "output")
        ],
        input_dims=input_dims,
        expected_output_dims=expected_output_dims)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ShouldRunTest(self, run_params):
    return (run_params.dynamic_engine and not trt_test.IsQuantizationMode(
        run_params.precision_mode)), "test dynamic engine and non-INT8"


if __name__ == "__main__":
  test.main()
