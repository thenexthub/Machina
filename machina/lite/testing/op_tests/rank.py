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
"""Test configs for rank."""
import machina as tf
from machina.lite.testing.zip_test_utils import create_tensor_data
from machina.lite.testing.zip_test_utils import make_zip_of_tests
from machina.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_rank_tests(options):
  """Make a set of tests to do rank."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[], [0], [1, 1, 1, 3], [2, 3, 4, 5], [5, 5], [10]],
  }]

  def build_graph(parameters):
    """Build the rank op testing graph."""
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"], name="input")
    out = tf.rank(input_value)
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
