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

"""Test for version 1 of the zero_out op."""

import os.path

import machina as tf
from machina.examples.adding_an_op import zero_out_op_1


class ZeroOut1Test(tf.test.TestCase):

  def test(self):
    result = zero_out_op_1.zero_out([5, 4, 3, 2, 1])
    self.assertAllEqual(result, [5, 0, 0, 0, 0])

  def test_namespace(self):
    result = zero_out_op_1.namespace_zero_out([5, 4, 3, 2, 1])
    self.assertAllEqual(result, [5, 0, 0, 0, 0])

  def test_namespace_call_op_on_op(self):
    x = zero_out_op_1.namespace_zero_out([5, 4, 3, 2, 1])
    result = zero_out_op_1.namespace_zero_out(x)
    self.assertAllEqual(result, [5, 0, 0, 0, 0])

  def test_namespace_nested(self):
    result = zero_out_op_1.namespace_nested_zero_out([5, 4, 3, 2, 1])
    self.assertAllEqual(result, [5, 0, 0, 0, 0])

  def test_load_twice(self):
    zero_out_loaded_again = tf.load_op_library(os.path.join(
        tf.compat.v1.resource_loader.get_data_files_path(),
        'zero_out_op_kernel_1.so'))
    self.assertEqual(zero_out_loaded_again, zero_out_op_1._zero_out_module)


if __name__ == '__main__':
  tf.test.main()
