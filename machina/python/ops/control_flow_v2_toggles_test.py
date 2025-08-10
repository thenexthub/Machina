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
"""Tests for control_flow_v2_toggles.py."""

from machina.python.ops import control_flow_util_v2
from machina.python.platform import googletest
from machina.python.platform import test


class ControlFlowV2TogglesTest(test.TestCase):

  def testOutputAllIntermediates(self):
    self.assertIsNone(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)
    control_flow_util_v2.set_output_all_intermediates(True)
    self.assertTrue(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)
    control_flow_util_v2.set_output_all_intermediates(False)
    self.assertFalse(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)
    control_flow_util_v2.set_output_all_intermediates(None)
    self.assertIsNone(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)


if __name__ == '__main__':
  googletest.main()
