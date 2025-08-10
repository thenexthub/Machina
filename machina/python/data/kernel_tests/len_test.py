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
"""Tests for `tf.data.Dataset.__len__()`."""
from absl.testing import parameterized

from machina.python.data.kernel_tests import test_base
from machina.python.data.ops import dataset_ops
from machina.python.framework import combinations
from machina.python.platform import test


class LenTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.eager_only_combinations())
  def testKnown(self):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    self.assertLen(ds, 10)

  @combinations.generate(test_base.eager_only_combinations())
  def testInfinite(self):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements).repeat()
    with self.assertRaisesRegex(TypeError, "infinite"):
      len(ds)

  @combinations.generate(test_base.eager_only_combinations())
  def testUnknown(self):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements).filter(lambda x: True)
    with self.assertRaisesRegex(TypeError, "unknown"):
      len(ds)

  @combinations.generate(test_base.graph_only_combinations())
  def testGraphMode(self):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    with self.assertRaisesRegex(
        TypeError,
        r"`tf.data.Dataset` only supports `len` in eager mode. Use "
        r"`tf.data.Dataset.cardinality\(\)` instead."):
      len(ds)


if __name__ == "__main__":
  test.main()
