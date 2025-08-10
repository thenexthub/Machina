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
"""Tests for memory leaks in remote eager execution."""

from machina.python.eager import def_function
from machina.python.eager import remote
from machina.python.eager import test
from machina.python.eager.memory_tests import memory_test_util
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.ops import array_ops
from machina.python.training import server_lib


class RemoteWorkerMemoryTest(test.TestCase):

  def __init__(self, method):
    super(RemoteWorkerMemoryTest, self).__init__(method)

    # used for remote worker tests
    self._cached_server = server_lib.Server.create_local_server()
    self._cached_server_target = self._cached_server.target[len("grpc://"):]

  def testMemoryLeakInLocalCopy(self):
    if not memory_test_util.memory_profiler_is_available():
      self.skipTest("memory_profiler required to run this test")

    remote.connect_to_remote_host(self._cached_server_target)

    # Run a function locally with the input on a remote worker and ensure we
    # do not leak a reference to the remote tensor.

    @def_function.function
    def local_func(i):
      return i

    def func():
      with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
        x = array_ops.zeros([1000, 1000], dtypes.int32)

      local_func(x)

    memory_test_util.assert_no_leak(
        func, num_iters=100, increase_threshold_absolute_mb=50)


if __name__ == "__main__":
  test.main()
