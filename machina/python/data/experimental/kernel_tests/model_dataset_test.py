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
"""Tests for the private `_ModelDataset` transformation."""
from absl.testing import parameterized

from machina.python.data.experimental.ops import testing
from machina.python.data.kernel_tests import test_base
from machina.python.data.ops import dataset_ops
from machina.python.data.ops import map_op
from machina.python.data.ops import options as options_lib
from machina.python.framework import combinations
from machina.python.framework import errors
from machina.python.platform import test


class ModelDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testAutotuneOption(self):
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.map(lambda x: x).apply(
        testing.assert_next(["Root"]))
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.autotune.enabled = True
    dataset = dataset.with_options(options)
    get_next = self.getNext(dataset)

    self.assertEqual(0, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testParallelMapWithAutotune(self):
    dataset = dataset_ops.Dataset.range(1000)
    dataset = map_op._ParallelMapDataset(  # pylint: disable=protected-access
        dataset,
        lambda x: x + 1,
        num_parallel_calls=1,
        deterministic=True,
        use_inter_op_parallelism=False)
    dataset = dataset.map(
        lambda x: x + 1, num_parallel_calls=dataset_ops.AUTOTUNE)
    next_element = self.getNext(dataset)
    self.evaluate(next_element())


if __name__ == "__main__":
  test.main()
