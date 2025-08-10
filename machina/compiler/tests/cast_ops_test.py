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
"""Tests lowering of tf.bitcast"""

from machina.compiler.tests import xla_test
from machina.python.eager import def_function
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.ops import array_ops
from machina.python.ops import math_ops
from machina.python.ops import random_ops
from machina.python.ops import control_flow_ops
from machina.python.ops import image_ops
from machina.python.ops import io_ops
from machina.python.platform import test


class CastOpsTest(xla_test.XLATestCase):

  def testBitcastToLarger(self):
    with ops.device('device:{}:0'.format(self.device)):

      def f(x):
        t = array_ops.bitcast(x, dtypes.float32)
        return math_ops.reduce_sum(t, axis=1)

      compiled_f = def_function.function(f, jit_compile=True)

      x = random_ops.random_normal([10, 10, 2], dtype=dtypes.float16)
      with ops.device(self.device):
        out = f(x)
        compiled_out = compiled_f(x)
        self.assertAllClose(out, compiled_out)
        # 10,10,2--(bitcast-convert)-->10,10--(reduce)-->10
        self.assertEqual(out.shape[0], 10)

      hlo = compiled_f.experimental_get_compiler_ir(x)(stage='hlo')
      self.assertIn('f32[10,10]{1,0} bitcast-convert(f16[10,10,2]{2,1,0}', hlo)

  def testBitcastToSmaller(self):
    pass


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
