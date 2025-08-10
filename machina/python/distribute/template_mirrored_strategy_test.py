###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Tuesday, March 25, 2025.                                            #
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
"""Tests for make_template used with MirroredStrategy."""
from machina.python.distribute import distribute_lib
from machina.python.distribute import mirrored_strategy
from machina.python.framework import ops
from machina.python.framework import test_util
from machina.python.ops import init_ops
from machina.python.ops import template
from machina.python.ops import variable_scope
from machina.python.ops import variables
from machina.python.platform import test


class TemplateMirroredStrategyTest(test.TestCase):

  @test_util.disable_tfrt("Strategy not supported yet.")
  def test_merge_call(self):
    with ops.Graph().as_default():
      # The test is testing a v1 only function.
      if not test.is_gpu_available():
        self.skipTest("No GPU available")

      def fn():
        var1 = variable_scope.get_variable(
            "var1", shape=[], initializer=init_ops.constant_initializer(21.))
        distribute_lib.get_replica_context().merge_call(lambda _: ())
        var2 = variable_scope.get_variable(
            "var2", shape=[], initializer=init_ops.constant_initializer(2.))
        return var1 * var2

      temp = template.make_template("my_template", fn)

      strategy = mirrored_strategy.MirroredStrategy(["/cpu:0", "/gpu:0"])
      out = strategy.experimental_local_results(
          strategy.run(temp))

      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual([42., 42.], self.evaluate(out))


if __name__ == "__main__":
  test.main()
