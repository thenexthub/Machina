###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, May 19, 2025.                                               #
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
"""Test cases for XLA devices."""

from machina.python.client import session as session_lib
from machina.python.eager import context
from machina.python.framework import config
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.ops import array_ops
from machina.python.platform import test


class XlaDeviceGpuTest(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(XlaDeviceGpuTest, self).__init__(method_name)
    context.context().enable_xla_devices()

  def testCopiesToAndFromGpuWork(self):
    """Tests that copies between GPU and XLA devices work."""
    if not config.list_physical_devices("GPU"):
      return

    with session_lib.Session() as sess:
      x = array_ops.placeholder(dtypes.float32, [2])
      with ops.device("GPU"):
        y = x * 2
      with ops.device("device:MACHINA_XLACPU:0"):
        z = y * y
      with ops.device("GPU"):
        w = y + z
      result = sess.run(w, {x: [1.5, 0.5]})
    self.assertAllClose(result, [12., 2.], rtol=1e-3)


if __name__ == "__main__":
  test.main()
