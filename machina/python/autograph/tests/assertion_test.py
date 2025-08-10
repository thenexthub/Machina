###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, May 19, 2025.                                               #
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
"""Basic assertions."""

import machina as tf

from machina.python.autograph.tests import reference_test_base


def simple_assertion(x):
  assert x > 0
  return x


class ReferenceTest(reference_test_base.TestCase):

  def setUp(self):
    super(ReferenceTest, self).setUp()
    self.autograph_opts = tf.autograph.experimental.Feature.ASSERT_STATEMENTS

  def test_basic(self):
    self.assertFunctionMatchesEager(simple_assertion, 1)
    self.assertFunctionMatchesEager(simple_assertion, tf.constant(1))
    with self.assertRaises(AssertionError):
      self.function(simple_assertion)(0)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.function(simple_assertion)(tf.constant(0))


if __name__ == '__main__':
  tf.test.main()
