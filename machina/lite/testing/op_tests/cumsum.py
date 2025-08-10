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
"""Test configs for cumsum."""
import machina as tf
from machina.lite.testing.zip_test_utils import create_tensor_data
from machina.lite.testing.zip_test_utils import make_zip_of_tests
from machina.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_cumsum_tests(options):
  """Make a set of tests to do cumsum."""

  test_parameters = [{
      "shape": [(3, 6), (8, 9, 7), (2, 4, 3, 5)],
      "dtype": [tf.int32, tf.int64, tf.float32],
      "axis": [0, 1],
      "exclusive": [True, False],
      "reverse": [True, False],
  }]

  def build_graph(parameters):
    """Build the cumsum op testing graph."""
    input1 = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], shape=parameters["shape"])
    out = tf.math.cumsum(
        input1,
        parameters["axis"],
        exclusive=parameters["exclusive"],
        reverse=parameters["reverse"])
    return [input1], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input1 = create_tensor_data(parameters["dtype"], parameters["shape"])
    return [input1], sess.run(outputs, feed_dict=dict(zip(inputs, [input1])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
