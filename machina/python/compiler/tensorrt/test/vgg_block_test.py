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

import os

import numpy as np

from machina.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.ops import array_ops
from machina.python.ops import nn
from machina.python.ops import nn_impl
from machina.python.ops import nn_ops
from machina.python.platform import test


class VGGBlockTest(trt_test.TfTrtIntegrationTestBase):
  """Single vgg layer test in TF-TRT conversion."""

  def GraphFn(self, x):
    dtype = x.dtype
    x, _, _ = nn_impl.fused_batch_norm(
        x, [1.0, 1.0], [0.0, 0.0],
        mean=[0.5, 0.5],
        variance=[1.0, 1.0],
        is_training=False)
    e = constant_op.constant(
        np.random.randn(1, 1, 2, 6), name="weights", dtype=dtype)
    conv = nn.conv2d(
        input=x, filter=e, strides=[1, 2, 2, 1], padding="SAME", name="conv")
    b = constant_op.constant(np.random.randn(6), name="bias", dtype=dtype)
    t = nn.bias_add(conv, b, name="biasAdd")
    relu = nn.relu(t, "relu")
    idty = array_ops.identity(relu, "ID")
    v = nn_ops.max_pool(
        idty, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
    return array_ops.squeeze(v, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[5, 8, 8, 2]],
                            [[5, 2, 2, 6]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_000"]

  # TODO(b/159459919): remove this routine to disallow native segment execution.
  def setUp(self):
    super().setUp()
    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"


if __name__ == "__main__":
  test.main()
