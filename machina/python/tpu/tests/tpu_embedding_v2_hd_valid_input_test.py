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
"""Tests for TPU Embeddings mid level API on TPU."""


import numpy as np

from machina.python.compat import v2_compat
from machina.python.data.ops import dataset_ops
from machina.python.distribute import distribute_lib
from machina.python.eager import def_function
from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.framework.tensor_shape import TensorShape
from machina.python.platform import test
from machina.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

  def test_enqueue_dense_sparse_ragged(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    dataset = self._create_high_dimensional_dense_dataset(strategy)
    dense_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    sparse = self._create_high_dimensional_sparse_dataset(strategy)
    sparse_iter = iter(
        strategy.experimental_distribute_dataset(
            sparse,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    ragged = self._create_high_dimensional_ragged_dataset(strategy)
    ragged_iter = iter(
        strategy.experimental_distribute_dataset(
            ragged,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    mid_level_api.build([
        TensorShape([self.batch_size, self.data_batch_size, 1]),
        TensorShape([self.batch_size, self.data_batch_size, 2]),
        TensorShape([self.batch_size, self.data_batch_size, 3])
    ])

    @def_function.function
    def test_fn():

      def step():
        return mid_level_api.dequeue()

      features = (next(dense_iter)[0], next(sparse_iter)[1],
                  next(ragged_iter)[2])
      mid_level_api.enqueue(features, training=False)
      return strategy.run(step)

    test_fn()

  def test_different_input_shapes(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_high_dimensional_sparse_dataset(strategy)
    sparse_iter = iter(
        strategy.experimental_distribute_dataset(
            sparse,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))
    # Create a feature with shape (1, 3, 1)
    dense_feature = constant_op.constant(
        np.zeros(3), shape=(1, 3, 1), dtype=dtypes.int32)
    dense_dataset = dataset_ops.DatasetV2.from_tensors(
        dense_feature).unbatch().repeat().batch(
            1 * strategy.num_replicas_in_sync, drop_remainder=True)
    dense_iter = iter(
        strategy.experimental_distribute_dataset(
            dense_dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def test_fn():

      def step():
        return mid_level_api.dequeue()

      features = (next(dense_iter), next(sparse_iter)[1], next(sparse_iter)[2])
      mid_level_api.enqueue(features, training=False)
      return strategy.run(step)

    test_fn()

    self.assertEqual(mid_level_api._output_shapes, [
        TensorShape((1, 3)),
        TensorShape((self.batch_size, self.data_batch_size)),
        TensorShape((self.batch_size, self.data_batch_size))
    ])

  def test_output_shapes_priority_over_feature_config_and_build(self):
    _, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    # The output shapes setting in the feature config has the first priority.
    mid_level_api._output_shapes = [TensorShape((2, 4)) for _ in range(3)]
    mid_level_api.build([TensorShape((2, None, None)) for _ in range(3)])
    self.assertEqual(mid_level_api._output_shapes,
                     [TensorShape((2, 4)) for _ in range(3)])

if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
