###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Friday, April 11, 2025.                                             #
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
"""Test DistributionStrategy in the zero batch case."""

from absl.testing import parameterized
import numpy as np

from machina.python.data.ops import dataset_ops
from machina.python.distribute import combinations
from machina.python.distribute import strategy_combinations
from machina.python.distribute import test_util
from machina.python.eager import backprop
from machina.python.eager import def_function
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.layers import normalization
from machina.python.ops import array_ops
from machina.python.ops import variables
from machina.python.ops.losses import losses
from machina.python.platform import test
from machina.python.training import gradient_descent


class NormalizationTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy,
          ],
          mode=["graph"],
          fused=[True, False]))
  def testBNWithZeroBatchInputGraph(self, distribution, fused):
    distribution.extended.experimental_enable_get_next_as_optional = True
    with distribution.scope(), self.cached_session() as sess:
      bn_list = []
      inputs = np.random.random((0, 4, 4, 3)) + 100
      targets = np.random.random((0, 4, 4, 3))
      inputs_placeholder = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 4, 4, 3])
      targets_placeholder = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 4, 4, 3])

      def step_fn(is_training, inputs, targets=None):
        bn = normalization.BatchNormalization(
            axis=3, epsilon=1e-3, momentum=0.9, fused=fused)
        bn_list.append(bn)
        outputs = bn.apply(inputs, training=is_training)
        if not is_training:
          return outputs

        loss = losses.mean_squared_error(targets, outputs)
        optimizer = gradient_descent.GradientDescentOptimizer(0.01)
        train_op = optimizer.minimize(loss)
        with ops.control_dependencies([train_op]):
          return array_ops.identity(loss)

      train_op = distribution.extended.call_for_each_replica(
          step_fn, args=(True, inputs_placeholder, targets_placeholder))
      predict_op = distribution.extended.call_for_each_replica(
          step_fn, args=(False, inputs_placeholder))
      bn = bn_list[0]

      self.evaluate(variables.global_variables_initializer())

      # Check for initial statistics and weights.
      moving_mean, moving_var = self.evaluate(
          [bn.moving_mean, bn.moving_variance])
      self.assertAllEqual([0, 0, 0], moving_mean)
      self.assertAllEqual([1, 1, 1], moving_var)

      np_gamma, np_beta = self.evaluate([bn.gamma, bn.beta])
      self.assertAllEqual([1, 1, 1], np_gamma)
      self.assertAllEqual([0, 0, 0], np_beta)

      for _ in range(100):
        np_output, _, _ = sess.run([train_op] + bn.updates, {
            inputs_placeholder: inputs,
            targets_placeholder: targets
        })
        self.assertEqual(0.0, np_output)

      # Verify that the statistics and weights are not changed after training.
      moving_mean, moving_var = self.evaluate(
          [bn.moving_mean, bn.moving_variance])
      self.assertAllEqual([0, 0, 0], moving_mean)
      self.assertAllEqual([1, 1, 1], moving_var)

      np_gamma, np_beta = self.evaluate([bn.gamma, bn.beta])
      self.assertAllEqual([1, 1, 1], np_gamma)
      self.assertAllEqual([0, 0, 0], np_beta)

      # Test inference.
      np_output = sess.run(predict_op, {inputs_placeholder: inputs})
      self.assertEqual([], np_output.tolist())

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy,
          ],
          mode=["eager"],
          fused=[True, False]))
  def testBNWithZeroBatchInput(self, distribution, fused):
    distribution.extended.experimental_enable_get_next_as_optional = True
    with distribution.scope():
      inputs = np.random.random((0, 4, 4, 3)).astype(np.float32) + 100
      targets = np.random.random((0, 4, 4, 3)).astype(np.float32)
      bn = normalization.BatchNormalization(
          axis=3, epsilon=1e-3, momentum=0.9, fused=fused)
      optimizer = gradient_descent.GradientDescentOptimizer(0.01)

      @def_function.function
      def train_step():
        def step_fn(inputs, targets):
          with backprop.GradientTape() as tape:
            outputs = bn.apply(inputs, training=True)
            loss = losses.mean_squared_error(targets, outputs)
          grads = tape.gradient(loss, bn.variables)
          optimizer.apply_gradients(zip(grads, bn.variables))
          return loss

        return distribution.run(step_fn, args=(inputs, targets))

      for _ in range(100):
        np_output = train_step().numpy()
        self.assertEqual(0.0, np_output)

      # Verify that the statistics and weights are not changed after training.
      self.assertAllEqual([0, 0, 0], bn.moving_mean.numpy())
      self.assertAllEqual([1, 1, 1], bn.moving_variance.numpy())
      self.assertAllEqual([1, 1, 1], bn.gamma.numpy())
      self.assertAllEqual([0, 0, 0], bn.beta.numpy())

      @def_function.function
      def test_step():
        def step_fn(inputs):
          outputs = bn.apply(inputs, training=False)
          return outputs

        return distribution.run(step_fn, args=(inputs,))

      # Test inference.
      self.assertAllEqual(np.zeros(shape=(0, 4, 4, 3), dtype=np.float32),
                          test_step().numpy())

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy,
          ],
          mode=["eager"],
          fused=[True, False]))
  def testBNWithDynamicBatchInputEager(self, distribution, fused):
    distribution.extended.experimental_enable_get_next_as_optional = True
    with distribution.scope():
      # Explicitly create dataset with drop_remainder=False.
      # This would make batch size unknown.
      inputs = np.random.random((11, 4, 4, 3)).astype(np.float32) + 100
      targets = np.random.random((11, 4, 4, 3)).astype(np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets)).batch(
          10, drop_remainder=False).repeat()
      dataset_iterator = iter(
          distribution.experimental_distribute_dataset(dataset))

      bn = normalization.BatchNormalization(
          axis=-1, epsilon=1e-3, momentum=0.9, fused=fused)
      optimizer = gradient_descent.GradientDescentOptimizer(0.01)

      @def_function.function
      def train_step(iterator):

        def step_fn(inputs):
          features, targets = inputs
          with backprop.GradientTape() as tape:
            outputs = bn(features, training=True)
            loss = losses.mean_squared_error(targets, outputs)

          grads = tape.gradient(loss, bn.variables)
          optimizer.apply_gradients(zip(grads, bn.variables))
          return loss

        return distribution.run(step_fn, args=(next(iterator),))

      for _ in range(100):
        train_step(dataset_iterator).numpy()

      # Verify that the statistics and weights are updated.
      self.assertNotAllEqual(np.ndarray([0, 0, 0]), bn.moving_mean.numpy())
      self.assertNotAllEqual(np.ndarray([1, 1, 1]), bn.moving_variance.numpy())
      self.assertNotAllEqual(np.ndarray([1, 1, 1]), bn.gamma.numpy())
      self.assertNotAllEqual(np.ndarray([0, 0, 0]), bn.beta.numpy())


if __name__ == "__main__":
  test_util.main()
