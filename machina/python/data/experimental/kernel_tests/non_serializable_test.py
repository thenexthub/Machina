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
"""Tests for `tf.data.experimental.non_serializable()`."""
from absl.testing import parameterized

from machina.python.data.experimental.ops import testing
from machina.python.data.kernel_tests import test_base
from machina.python.data.ops import dataset_ops
from machina.python.data.ops import options as options_lib
from machina.python.framework import combinations
from machina.python.platform import test


class NonSerializableTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testNonSerializable(self):
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.apply(testing.assert_next(["FiniteSkip"]))
    dataset = dataset.skip(0)  # Should not be removed by noop elimination
    dataset = dataset.apply(testing.non_serializable())
    dataset = dataset.apply(testing.assert_next(["MemoryCacheImpl"]))
    dataset = dataset.skip(0)  # Should be removed by noop elimination
    dataset = dataset.cache()
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])

  @combinations.generate(test_base.default_test_combinations())
  def testNonSerializableAsDirectInput(self):
    """Tests that non-serializable dataset can be OptimizeDataset's input."""
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.apply(testing.non_serializable())
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])


if __name__ == "__main__":
  test.main()
