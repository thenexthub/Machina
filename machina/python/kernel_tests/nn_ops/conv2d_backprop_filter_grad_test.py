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
"""Tests for convolution related functionality in machina.ops.nn."""

import unittest
import numpy as np

from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.framework import test_util
from machina.python.ops import array_ops
from machina.python.ops import gradient_checker
from machina.python.ops import nn_ops
import machina.python.ops.nn_grad  # pylint: disable=unused-import
from machina.python.platform import test


@test_util.run_all_without_tensor_float_32(
    "Run Conv2D backprop without TF32 on GPU")
class Conv2DBackpropFilterGradTest(test.TestCase):

  # TODO(b/292002914): Enable this test after fixing its flakyness.
  @unittest.skip("Disable the flaky test.")
  @test_util.run_deprecated_v1
  def testGradient(self):
    with self.cached_session():
      for padding in [
          "SAME",
          "VALID",
          [(0, 0), (1, 2), (3, 4), (0, 0)],
          [(0, 0), (0, 3), (4, 2), (0, 0)]
      ]:
        for stride in [1, 2]:
          np.random.seed(1)
          in_shape = [5, 8, 6, 4]
          in_val = constant_op.constant(
              2 * np.random.random_sample(in_shape) - 1, dtype=dtypes.float32)
          filter_shape = [3, 3, 4, 6]
          # Make a convolution op with the current settings, just to easily get
          # the shape of the output.
          conv_out = nn_ops.conv2d(
              in_val,
              array_ops.zeros(filter_shape),
              strides=[1, stride, stride, 1],
              padding=padding)
          out_backprop_shape = conv_out.get_shape().as_list()
          out_backprop_val = constant_op.constant(
              2 * np.random.random_sample(out_backprop_shape) - 1,
              dtype=dtypes.float32)
          output = nn_ops.conv2d_backprop_filter(
              in_val,
              filter_shape,
              out_backprop_val,
              strides=[1, stride, stride, 1],
              padding=padding)
          err = gradient_checker.compute_gradient_error(
              [in_val, out_backprop_val], [in_shape, out_backprop_shape],
              output, filter_shape)
          print("conv2d_backprop_filter gradient err = %g " % err)
          err_tolerance = 3e-2 if test.is_gpu_available() else 2e-3
          self.assertLess(
              err,
              err_tolerance,
              msg="padding={0},stride={1},".format(str(padding), stride))

  @test_util.run_deprecated_v1
  def testGradientDilatedConv(self):
    if test.is_gpu_available(cuda_only=True):
      with self.session():
        for padding in [
            "SAME",
            "VALID",
            [(0, 0), (3, 5), (2, 1), (0, 0)],
            [(0, 0), (5, 2), (5, 1), (0, 0)]
        ]:
          for stride in [1, 2]:
            np.random.seed(1)
            in_shape = [5, 8, 6, 4]
            in_val = constant_op.constant(
                2 * np.random.random_sample(in_shape) - 1, dtype=dtypes.float32)
            filter_shape = [3, 3, 4, 6]
            # Make a convolution op with the current settings,
            # just to easily get the shape of the output.
            conv_out = nn_ops.conv2d(
                in_val,
                array_ops.zeros(filter_shape),
                dilations=[1, 2, 2, 1],
                strides=[1, stride, stride, 1],
                padding=padding)
            out_backprop_shape = conv_out.get_shape().as_list()
            out_backprop_val = constant_op.constant(
                2 * np.random.random_sample(out_backprop_shape) - 1,
                dtype=dtypes.float32)
            output = nn_ops.conv2d_backprop_filter(
                in_val,
                filter_shape,
                out_backprop_val,
                dilations=[1, 2, 2, 1],
                strides=[1, stride, stride, 1],
                padding=padding)
            err = gradient_checker.compute_gradient_error(
                [in_val, out_backprop_val], [in_shape, out_backprop_shape],
                output, filter_shape)
            print("conv2d_backprop_filter gradient err = %g " % err)
            err_tolerance = 1e-2
            self.assertLess(err, err_tolerance)


if __name__ == "__main__":
  test.main()
