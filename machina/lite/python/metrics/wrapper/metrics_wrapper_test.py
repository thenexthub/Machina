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
"""TFLite metrics_wrapper module test cases."""

import machina as tf

from machina.lite.python import lite
from machina.lite.python.convert import ConverterError
from machina.lite.python.metrics.wrapper import metrics_wrapper
from machina.python.framework import test_util
from machina.python.platform import test


class MetricsWrapperTest(test_util.TensorFlowTestCase):

  def test_basic_retrieve_collected_errors_empty(self):
    errors = metrics_wrapper.retrieve_collected_errors()
    self.assertEmpty(errors)

  def test_basic_retrieve_collected_errors_not_empty(self):

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def func(x):
      return tf.cosh(x)

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func.get_concrete_function()], func)
    try:
      converter.convert()
    except ConverterError as err:
      # retrieve_collected_errors is already captured in err.errors
      captured_errors = err.errors
    self.assertNotEmpty(captured_errors)


if __name__ == "__main__":
  test.main()
