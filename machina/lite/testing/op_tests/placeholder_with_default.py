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
"""Test configs for placeholder_with_default."""
import numpy as np
import machina as tf

from machina.lite.testing.zip_test_utils import make_zip_of_tests
from machina.lite.testing.zip_test_utils import MAP_TF_TO_NUMPY_TYPE
from machina.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_placeholder_with_default_tests(options):
  """Make a set of tests to test placeholder_with_default."""

  test_parameters = [{
      "dtype": [tf.float32, tf.int32, tf.int64],
  }]

  def build_graph(parameters):
    """Build the placeholder_with_default testing graph."""
    const_node = tf.constant([1, 2, 2, 0],
                             shape=[2, 2],
                             dtype=parameters["dtype"])
    input_tensor = tf.compat.v1.placeholder_with_default(
        const_node, shape=[2, 2], name="input")
    out = tf.equal(input_tensor, const_node, name="output")

    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    numpy_type = MAP_TF_TO_NUMPY_TYPE[parameters["dtype"]]
    input_value = np.array([[1, 0], [2, 1]], numpy_type)
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
