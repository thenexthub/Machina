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
"""Tests for machina.kernels.logging_ops."""

import sys

from machina.python.framework import test_util
from machina.python.ops import logging_ops
from machina.python.ops import math_ops
from machina.python.platform import test
from machina.python.platform import tf_logging


class PrintV2LoggingLevelTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testPrintOneTensorLogInfo(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(
            tensor, output_stream=tf_logging.info)
        self.evaluate(print_op)
      self.assertTrue("I" in printed.contents())
      expected = "[0 1 2 ... 7 8 9]"
      self.assertTrue(expected in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintOneTensorLogWarning(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(
            tensor, output_stream=tf_logging.warning)
        self.evaluate(print_op)
      self.assertTrue("W" in printed.contents())
      expected = "[0 1 2 ... 7 8 9]"
      self.assertTrue(expected in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def testPrintOneTensorLogError(self):
    with self.cached_session():
      tensor = math_ops.range(10)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(
            tensor, output_stream=tf_logging.error)
        self.evaluate(print_op)
      self.assertTrue("E" in printed.contents())
      expected = "[0 1 2 ... 7 8 9]"
      self.assertTrue(expected in printed.contents())


if __name__ == "__main__":
  test.main()
