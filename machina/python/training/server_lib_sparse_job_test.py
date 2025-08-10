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
"""Tests for tf.GrpcServer."""

from machina.python.client import session
from machina.python.framework import constant_op
from machina.python.framework import ops
from machina.python.framework import test_util
from machina.python.platform import test
from machina.python.training import server_lib


class SparseJobTest(test.TestCase):

  # TODO(b/34465411): Starting multiple servers with different configurations
  # in the same test is flaky. Move this test case back into
  # "server_lib_test.py" when this is no longer the case.
  @test_util.run_deprecated_v1
  def testSparseJob(self):
    server = server_lib.Server({"local": {37: "localhost:0"}})
    with ops.device("/job:local/task:37"):
      a = constant_op.constant(1.0)

    with session.Session(server.target) as sess:
      self.assertEqual(1.0, self.evaluate(a))


if __name__ == "__main__":
  test.main()
