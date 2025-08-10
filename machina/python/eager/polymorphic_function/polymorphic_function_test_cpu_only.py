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

from absl.testing import parameterized

from machina.python.eager.polymorphic_function import polymorphic_function
from machina.python.framework import errors
from machina.python.framework import ops
from machina.python.framework import test_util
from machina.python.platform import test


class FunctionCpuOnlyTest(test.TestCase, parameterized.TestCase):
  """Test that jit_compile=True correctly throws an exception if XLA is not available.

  This test should only be run without `--config=cuda`, as that implicitly links
  in XLA JIT.
  """

  def testJitCompileRaisesExceptionWhenXlaIsUnsupported(self):
    if test.is_built_with_rocm() or test_util.is_xla_enabled():
      return

    with self.assertRaisesRegex(errors.UnimplementedError,
                                'support for that platform linked in'):

      @polymorphic_function.function(jit_compile=True)
      def fn(x):
        return x + x

      fn([1, 1, 2, 3])


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
