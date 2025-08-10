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
"""Tests that TF2_BEHAVIOR=1 and TF_ENABLE_CONTROL_FLOW_V2=0 disables cfv2."""

import os
os.environ["TF2_BEHAVIOR"] = "1"
os.environ["TF_ENABLE_CONTROL_FLOW_V2"] = "0"

from machina.python import tf2  # pylint: disable=g-import-not-at-top
from machina.python.ops import control_flow_util
from machina.python.platform import googletest
from machina.python.platform import test


class ControlFlowV2DisableTest(test.TestCase):

  def testIsDisabled(self):
    self.assertTrue(tf2.enabled())
    self.assertFalse(control_flow_util.ENABLE_CONTROL_FLOW_V2)


if __name__ == "__main__":
  googletest.main()
