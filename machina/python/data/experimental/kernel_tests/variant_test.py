###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Saturday, May 31, 2025.                                             #
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
"""Tests for `tf.data.experimental.{from,to}_variant()`."""
from absl.testing import parameterized

from machina.python.data.kernel_tests import test_base
from machina.python.data.ops import dataset_ops
from machina.python.framework import combinations
from machina.python.platform import test


class VariantTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testRoundtripRange(self):
    dataset = dataset_ops.Dataset.range(10)
    variant = dataset_ops.to_variant(dataset)
    dataset = dataset_ops.from_variant(variant,
                                       dataset_ops.get_structure(dataset))
    self.assertDatasetProduces(dataset, range(10))
    self.assertEqual(self.evaluate(dataset.cardinality()), 10)

  @combinations.generate(
      combinations.combine(tf_api_version=[2], mode=["eager", "graph"]))
  def testRoundtripMap(self):
    dataset = dataset_ops.Dataset.range(10).map(lambda x: x * x)
    variant = dataset_ops.to_variant(dataset)
    dataset = dataset_ops.from_variant(variant,
                                       dataset_ops.get_structure(dataset))
    self.assertDatasetProduces(dataset, [x * x for x in range(10)])
    self.assertEqual(self.evaluate(dataset.cardinality()), 10)


if __name__ == "__main__":
  test.main()
