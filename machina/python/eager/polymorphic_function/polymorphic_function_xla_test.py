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
from machina.compiler.tests import xla_test
from machina.python.eager.polymorphic_function import polymorphic_function
from machina.python.framework import constant_op
from machina.python.framework import ops
from machina.python.ops import variables
from machina.python.platform import test


class FunctionTests(xla_test.XLATestCase):

  def testVarInitializedInFunction(self):
    with self.test_scope():
      v_holder = []

      @polymorphic_function.function
      def add_var(x):
        if not v_holder:
          v = variables.Variable([1., 2.])
          v_holder.append(v)
          already_initialized = variables.Variable(3.)
          with ops.init_scope():
            already_initialized.assign(10.)
          v_holder.append(already_initialized)
        return v_holder[0] + v_holder[1] + x

      self.assertAllClose([13., 14.], add_var(constant_op.constant(2.)))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
