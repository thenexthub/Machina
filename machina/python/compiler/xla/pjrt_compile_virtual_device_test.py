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
"""Tests virtual device compilation + execution using the Device API (aka PjRt).

This feature is still under active development and is protected behind the
`--tf_xla_use_device_api` flag in the `TF_MACHINA_MACHINA_XLA_FLAGS` environment variable.
"""
from machina.python.eager import context
from machina.python.eager import def_function
from machina.python.eager import test
from machina.python.framework import config
from machina.python.framework import constant_op
from machina.python.framework import ops
from machina.python.ops import variables


class PjrtCompileVirtualDeviceTest(test.TestCase):

  def setUp(self):
    super().setUp()
    gpus = config.list_physical_devices("GPU")
    config.set_logical_device_configuration(
        gpus[0],
        [
            context.LogicalDeviceConfiguration(memory_limit=1024),
            context.LogicalDeviceConfiguration(memory_limit=1024),
            context.LogicalDeviceConfiguration(memory_limit=1024),
        ],
    )

  def test_xla_launch_and_tf_kernel_on_gpu_device(self):
    @def_function.function(jit_compile=True)
    def foo(x, y):
      return x + y + 1

    @def_function.function(jit_compile=True)
    def bar(x, y):
      x.assign(y)
      y.assign_add([1.0, 1.0])

    with ops.device("/device:GPU:1"):
      a = constant_op.constant([1.0, 2.0])
      x = variables.Variable([0.0, 1.0])
      result_tensor = foo(x, a)

    self.assertAllClose(result_tensor.numpy(), [2.0, 4.0], atol=1e-05)

    # The following use case is tested:
    # Variable updates following an XLA computation that reads the updated
    # variables.
    with ops.device("/device:GPU:1"):
      var_a = variables.Variable([0.0, 1.0])
      var_b = variables.Variable([1.0, 2.0])
      bar(var_a, var_b)
      result = foo(var_a, var_b)

    self.assertAllClose([1.0, 2.0], var_a.value(), atol=1e-05)
    self.assertAllClose([2.0, 3.0], var_b.value(), atol=1e-05)
    self.assertAllClose(result, [4.0, 6.0], atol=1e-05)


if __name__ == "__main__":
  test.main()
