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
###############################################################################
"""Test configs for pack."""
import numpy as np
import machina as tf
from machina.lite.testing.zip_test_utils import create_tensor_data
from machina.lite.testing.zip_test_utils import make_zip_of_tests
from machina.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_pack_tests(options):
  """Make a set of tests to do stack."""

  test_parameters = [
      # Avoid creating all combinations to keep the test size small.
      {
          "dtype": [tf.float32],
          "base_shape": [[3, 4, 3], [3, 4], [5]],
          "num_tensors": [1, 2, 3, 4, 5, 6],
          "axis": [0, 1, 2, 3],
          "additional_shape": [1, 2, 3],
          "fully_quantize": [False],
      },
      {
          "dtype": [tf.int32],
          "base_shape": [[3, 4, 3], [3, 4], [5]],
          "num_tensors": [6],
          "axis": [0, 1, 2, 3],
          "additional_shape": [1, 2, 3],
          "fully_quantize": [False],
      },
      {
          "dtype": [tf.int64],
          "base_shape": [[3, 4, 3], [3, 4], [5]],
          "num_tensors": [5],
          "axis": [0, 1, 2, 3],
          "additional_shape": [1, 2, 3],
          "fully_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "base_shape": [[1, 4, 3], [3, 4], [5]],
          "num_tensors": [2, 3, 4, 5, 6],  # 1 tensor would go to Reshape.
          "axis": [0, 1, 2, 3],
          "additional_shape": [1, 2, 3],
          "fully_quantize": [True],
      },
  ]

  def get_shape(parameters):
    """Return a tweaked version of 'base_shape'."""
    axis = parameters["axis"]
    shape = parameters["base_shape"][:]
    if axis < len(shape):
      shape[axis] += parameters["additional_shape"]
    return shape

  def build_graph(parameters):
    all_tensors = []
    for n in range(0, parameters["num_tensors"]):
      input_tensor = tf.compat.v1.placeholder(
          dtype=parameters["dtype"],
          name=("input%d" % n),
          shape=get_shape(parameters))
      all_tensors.append(input_tensor)
    out = tf.stack(all_tensors, parameters["axis"])
    return all_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    all_values = []
    for _ in range(0, parameters["num_tensors"]):
      input_values = create_tensor_data(
          np.float32, get_shape(parameters), min_value=-1, max_value=1)
      all_values.append(input_values)
    return all_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, all_values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=117)
