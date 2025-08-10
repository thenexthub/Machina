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
"""Tests for Proximal Gradient Descent optimizer."""

import numpy as np

from machina.compiler.tests import xla_test
from machina.python.framework import constant_op
from machina.python.ops import resource_variable_ops
from machina.python.ops import variables
from machina.python.platform import test
from machina.python.training import gradient_descent
from machina.python.training import proximal_gradient_descent


class ProximalGradientDescentOptimizerTest(xla_test.XLATestCase):

  def testResourceProximalGradientDescentwithoutRegularization(self):
    with self.session(), self.test_scope():
      var0 = resource_variable_ops.ResourceVariable([0.0, 0.0])
      var1 = resource_variable_ops.ResourceVariable([0.0, 0.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])
      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      self.assertAllClose([0.0, 0.0], self.evaluate(var0))
      self.assertAllClose([0.0, 0.0], self.evaluate(var1))

      # Run 3 steps Proximal Gradient Descent.
      for _ in range(3):
        update.run()

      self.assertAllClose(np.array([-0.9, -1.8]), self.evaluate(var0))
      self.assertAllClose(np.array([-0.09, -0.18]), self.evaluate(var1))

  def testProximalGradientDescentwithoutRegularization2(self):
    with self.session(), self.test_scope():
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
      var1 = resource_variable_ops.ResourceVariable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([4.0, 3.0], self.evaluate(var1))

      # Run 3 steps Proximal Gradient Descent
      for _ in range(3):
        update.run()

      self.assertAllClose(np.array([0.1, 0.2]), self.evaluate(var0))
      self.assertAllClose(np.array([3.91, 2.82]), self.evaluate(var1))

  def testProximalGradientDescentWithL1(self):
    with self.session(), self.test_scope():
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
      var1 = resource_variable_ops.ResourceVariable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.001, l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([4.0, 3.0], self.evaluate(var1))

      # Run 10 steps proximal gradient descent.
      for _ in range(10):
        update.run()

      self.assertAllClose(np.array([-1.988, -3.988001]), self.evaluate(var0))
      self.assertAllClose(np.array([3.67, 2.37]), self.evaluate(var1))

  def testProximalGradientDescentWithL1_L2(self):
    with self.session(), self.test_scope():
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
      var1 = resource_variable_ops.ResourceVariable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_gradient_descent.ProximalGradientDescentOptimizer(
          3.0, l1_regularization_strength=0.001, l2_regularization_strength=2.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([4.0, 3.0], self.evaluate(var1))

      # Run 10 steps Proximal Gradient Descent
      for _ in range(10):
        update.run()

      self.assertAllClose(np.array([-0.0495, -0.0995]), self.evaluate(var0))
      self.assertAllClose(np.array([-0.0045, -0.0095]), self.evaluate(var1))

  def applyOptimizer(self, opt, steps=5):
    var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
    var1 = resource_variable_ops.ResourceVariable([3.0, 4.0])
    grads0 = constant_op.constant([0.1, 0.2])
    grads1 = constant_op.constant([0.01, 0.02])

    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    self.evaluate(variables.global_variables_initializer())

    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
    self.assertAllClose([3.0, 4.0], self.evaluate(var1))

    # Run ProximalAdagrad for a few steps
    for _ in range(steps):
      update.run()

    return self.evaluate(var0), self.evaluate(var1)

  def testEquivGradientDescentwithoutRegularization(self):
    with self.session(), self.test_scope():
      val0, val1 = self.applyOptimizer(
          proximal_gradient_descent.ProximalGradientDescentOptimizer(
              3.0,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0))

    with self.session(), self.test_scope():
      val2, val3 = self.applyOptimizer(
          gradient_descent.GradientDescentOptimizer(3.0))

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)


if __name__ == "__main__":
  test.main()
