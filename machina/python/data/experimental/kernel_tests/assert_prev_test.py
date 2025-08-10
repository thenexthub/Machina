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
"""Tests for `tf.data.experimental.assert_prev()`."""
from absl.testing import parameterized

from machina.python.data.experimental.ops import testing
from machina.python.data.kernel_tests import test_base
from machina.python.data.ops import dataset_ops
from machina.python.data.ops import options as options_lib
from machina.python.framework import combinations
from machina.python.framework import errors
from machina.python.platform import test


class AssertPrevTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testAssertPrev(self):
    dataset = dataset_ops.Dataset.from_tensors(0).map(
        lambda x: x, deterministic=True, num_parallel_calls=8).apply(
            testing.assert_prev([("ParallelMapDataset",
                                  {"deterministic", "true"})]))
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])

  @combinations.generate(test_base.default_test_combinations())
  def testIgnoreVersionSuffix(self):
    # The `batch` transformation creates a "BatchV2" dataset, but we should
    # still match that with "Batch".
    dataset = dataset_ops.Dataset.from_tensors(0).map(
        lambda x: x, deterministic=True, num_parallel_calls=8).batch(1).apply(
            testing.assert_prev([("BatchDataset", {}),
                                 ("ParallelMapDataset", {
                                     "deterministic": "true"
                                 })]))
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[[0]])

  @combinations.generate(test_base.default_test_combinations())
  def testAssertPrevInvalid(self):
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_prev([("Whoops", {})]))
    self.assertDatasetProduces(
        dataset,
        expected_error=(errors.InvalidArgumentError,
                        "Asserted transformation matching 'Whoops'"))

  @combinations.generate(test_base.default_test_combinations())
  def testAssertPrevShort(self):
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_prev([("TensorDataset", {}), ("Whoops", {})]))
    self.assertDatasetProduces(
        dataset,
        expected_error=(
            errors.InvalidArgumentError,
            "Asserted previous 2 transformations but encountered only 1."))

  @combinations.generate(test_base.default_test_combinations())
  def testAssertBadAttributeName(self):
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_prev([("TensorDataset", {
            "whoops": "true"
        })]))
    self.assertDatasetProduces(
        dataset,
        expected_error=(errors.InvalidArgumentError, "found no such attribute"))


if __name__ == "__main__":
  test.main()
