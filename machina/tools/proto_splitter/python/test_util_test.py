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
"""Tests for test_util."""

from absl.testing import parameterized

from machina.python.framework import dtypes
from machina.python.platform import test
from machina.tools.proto_splitter.python import test_util


class MakeGraphDefTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Float64", dtypes.float64), ("Float32", dtypes.float32)
  )
  def testMakeGraphDef(self, dtype):
    expected_sizes = [75, 50, 100, 95, 120]
    fn1 = [121, 153, 250, 55]
    fn2 = [552, 45]
    graph_def = test_util.make_graph_def_with_constant_nodes(
        expected_sizes, dtype=dtype, fn1=fn1, fn2=fn2)
    self.assertAllClose(
        expected_sizes, [node.ByteSize() for node in graph_def.node], atol=5
    )
    self.assertAllClose(
        fn1,
        [node.ByteSize() for node in graph_def.library.function[0].node_def],
        atol=10,
    )
    self.assertAllClose(
        fn2,
        [node.ByteSize() for node in graph_def.library.function[1].node_def],
        atol=10,
    )


if __name__ == "__main__":
  test.main()
