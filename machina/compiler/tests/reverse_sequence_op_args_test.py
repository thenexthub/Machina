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
"""Tests for machina.ops.reverse_sequence_op."""

from machina.compiler.tests import xla_test
from machina.python.compat import v2_compat
from machina.python.eager import def_function
from machina.python.framework import errors
from machina.python.ops import array_ops
from machina.python.platform import test


class ReverseSequenceArgsTest(xla_test.XLATestCase):
  """Tests argument verification of array_ops.reverse_sequence."""

  def testInvalidArguments(self):
    # seq_axis negative
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError), "seq_dim must be >=0"
    ):

      @def_function.function(jit_compile=True)
      def f(x):
        return array_ops.reverse_sequence(x, [2, 2], seq_axis=-1)

      f([[1, 2], [3, 4]])

    # batch_axis negative
    with self.assertRaisesRegex(ValueError, "batch_dim must be >=0"):

      @def_function.function(jit_compile=True)
      def g(x):
        return array_ops.reverse_sequence(x, [2, 2], seq_axis=1, batch_axis=-1)

      g([[1, 2], [3, 4]])


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
