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
"""Tests for functional operations."""

from machina.python.eager import def_function
from machina.python.framework import dtypes
from machina.python.framework import function
from machina.python.framework import ops
from machina.python.framework import sparse_tensor
from machina.python.framework import tensor_spec
from machina.python.ops import functional_ops
from machina.python.platform import test


class FunctionalOpsTest(test.TestCase):

  def testIfWithDefun(self):
    # Defun should only be used in graph mode
    with ops.Graph().as_default():
      @function.Defun(dtypes.float32)
      def Then(x):
        return x + 1

      @function.Defun(dtypes.float32)
      def Else(x):
        return x - 1

      inputs = [10.]
      result = self.evaluate(functional_ops.If(False, inputs, Then, Else))
      self.assertEqual([9.0], result)

  def testIfWithFunction(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def Then(x):
      return x + 1

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def Else(x):
      return x - 1

    inputs = [10.]
    then_cf = Then.get_concrete_function()
    else_cf = Else.get_concrete_function()
    result = self.evaluate(functional_ops.If(False, inputs, then_cf, else_cf))
    self.assertEqual([9.0], result)

  def testIfWithFunctionComposite(self):

    signature = [tensor_spec.TensorSpec([], dtypes.float32)]
    @def_function.function(input_signature=signature)
    def Then(x):
      return sparse_tensor.SparseTensor([[0]], [x + 1], [1])

    @def_function.function(input_signature=signature)
    def Else(x):
      return sparse_tensor.SparseTensor([[0]], [x - 1], [1])

    inputs = [10.]
    then_cf = Then.get_concrete_function()
    else_cf = Else.get_concrete_function()
    result = functional_ops.If(False, inputs, then_cf, else_cf)
    self.assertIsInstance(result, sparse_tensor.SparseTensor)
    self.assertAllEqual([9.0], result.values)


if __name__ == '__main__':
  test.main()
