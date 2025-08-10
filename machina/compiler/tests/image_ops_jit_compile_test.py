###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Saturday, May 31, 2025.                                             #
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
###############################################################################=
"""Tests for machina.ops.array_ops.repeat."""

from machina.compiler.tests import xla_test
from machina.python.eager import backprop
from machina.python.eager import def_function
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.ops import array_ops
from machina.python.ops import image_ops
from machina.python.ops import math_ops
from machina.python.ops import variables
from machina.python.platform import test


class ImageOpsTest(xla_test.XLATestCase):

  def testGradImageResize(self):
    """Tests that the gradient of image.resize is compilable."""
    with ops.device("device:{}:0".format(self.device)):
      img_width = 2048
      var = variables.Variable(array_ops.ones(1, dtype=dtypes.float32))

      def model(x):
        x = var * x
        x = image_ops.resize_images(
            x,
            size=[img_width, img_width],
            method=image_ops.ResizeMethod.BILINEAR)
        return x

      def train(x, y):
        with backprop.GradientTape() as tape:
          output = model(x)
          loss_value = math_ops.reduce_mean((y - output)**2)
        grads = tape.gradient(loss_value, [var])
        return grads

      compiled_train = def_function.function(train, jit_compile=True)
      x = array_ops.zeros((1, img_width // 2, img_width // 2, 1),
                          dtype=dtypes.float32)
      y = array_ops.zeros((1, img_width, img_width, 1), dtype=dtypes.float32)
      self.assertAllClose(train(x, y), compiled_train(x, y))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
