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
"""TensorFlow Lite Python metrics helpr TFLiteMetrics check."""
from machina.lite.python.metrics import metrics
from machina.python.framework import test_util
from machina.python.platform import test


class MetricsPortableTest(test_util.TensorFlowTestCase):

  def test_TFLiteMetrics_creation_success(self):
    metrics.TFLiteMetrics()

  def test_debugger_creation_counter_increase_success(self):
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_debugger_creation()

  def test_interpreter_creation_counter_increase_success(self):
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_interpreter_creation()

  def test_converter_attempt_counter_increase_success(self):
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_converter_attempt()

  def test_converter_success_counter_increase_success(self):
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_converter_success()

  def test_converter_params_set_success(self):
    stub = metrics.TFLiteMetrics()
    stub.set_converter_param('name', 'value')


if __name__ == '__main__':
  test.main()
