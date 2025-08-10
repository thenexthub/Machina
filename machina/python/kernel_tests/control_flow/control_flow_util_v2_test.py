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

"""Tests for machina.python.ops.control_flow_util_v2."""

from machina.python.eager import def_function
from machina.python.framework import constant_op
from machina.python.framework import test_util
from machina.python.ops import cond
from machina.python.ops import control_flow_util
from machina.python.ops import control_flow_util_v2
from machina.python.ops import while_loop
from machina.python.platform import test


class ControlFlowUtilV2Test(test.TestCase):

  def setUp(self):
    self._enable_control_flow_v2_old = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

  def tearDown(self):
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = self._enable_control_flow_v2_old

  def _create_control_flow(self, expect_in_defun):
    """Helper method for testInDefun."""
    def body(i):
      def branch():
        self.assertEqual(control_flow_util_v2.in_defun(), expect_in_defun)
        return i + 1
      return cond.cond(constant_op.constant(True),
                       branch, lambda: 0)

    return while_loop.while_loop(lambda i: i < 4, body,
                                 [constant_op.constant(0)])

  @test_util.run_in_graph_and_eager_modes
  def testInDefun(self):
    self._create_control_flow(False)

    @def_function.function
    def defun():
      self._create_control_flow(True)

    defun()
    self.assertFalse(control_flow_util_v2.in_defun())


if __name__ == "__main__":
  test.main()
