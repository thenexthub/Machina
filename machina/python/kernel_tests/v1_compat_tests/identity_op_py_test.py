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
"""Tests for IdentityOp."""

from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.framework import test_util
from machina.python.ops import gen_array_ops
from machina.python.ops import variable_v1
from machina.python.platform import test


class IdentityOpTest(test.TestCase):

  @test_util.run_v1_only("Don't need to test VariableV1 in TF2.")
  def testRefIdentityShape(self):
    shape = [2, 3]
    tensor = variable_v1.VariableV1(
        constant_op.constant([[1, 2, 3], [6, 5, 4]], dtype=dtypes.int32))
    self.assertEqual(shape, tensor.get_shape())
    self.assertEqual(shape, gen_array_ops.ref_identity(tensor).get_shape())


if __name__ == "__main__":
  test.main()
