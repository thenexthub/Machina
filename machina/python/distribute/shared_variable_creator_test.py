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
"""Tests for SharedVariableCreator."""

from machina.python.distribute import shared_variable_creator
from machina.python.eager import test
from machina.python.framework import test_util
from machina.python.ops import variable_scope
from machina.python.ops import variable_v1


class CanonicalizeVariableNameTest(test.TestCase):

  def _canonicalize(self, name):
    return shared_variable_creator._canonicalize_variable_name(name)

  def testNoName(self):
    self.assertEqual("Variable", self._canonicalize(None))

  def testPatternInMiddle(self):
    self.assertEqual("foo/bar/baz", self._canonicalize("foo_1/bar_1/baz"))

  def testPatternAtEnd(self):
    self.assertEqual("foo", self._canonicalize("foo_1"))

  def testWrongPatterns(self):
    self.assertEqual("foo_1:0", self._canonicalize("foo_1:0"))
    self.assertEqual("foo1", self._canonicalize("foo1"))
    self.assertEqual("foo_a", self._canonicalize("foo_a"))


class SharedVariableCreatorTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testSharedVariable(self):

    shared_variable_store = {}
    num_devices = 3
    creator_fns = []
    for i in range(num_devices):
      creator_fn = shared_variable_creator.make_fn(shared_variable_store, i)
      creator_fns.append(creator_fn)

    with variable_scope.variable_creator_scope(creator_fns[0]):
      v0 = variable_v1.VariableV1(1.0, name="foo")

    with variable_scope.variable_creator_scope(creator_fns[1]):
      v1 = variable_v1.VariableV1(1.0, name="foo")

    with variable_scope.variable_creator_scope(creator_fns[2]):
      v2 = variable_v1.VariableV1(1.0, name="foo")

    # v1 and v2 should be same as v0
    self.assertIs(v1, v0)
    self.assertIs(v2, v0)


if __name__ == "__main__":
  test.main()
