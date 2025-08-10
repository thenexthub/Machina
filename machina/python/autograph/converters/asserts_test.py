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
"""Tests for asserts module."""

from machina.python.autograph.converters import asserts
from machina.python.autograph.converters import functions
from machina.python.autograph.converters import return_statements
from machina.python.autograph.core import converter_testing
from machina.python.framework import constant_op
from machina.python.framework import errors_impl
from machina.python.platform import test


class AssertsTest(converter_testing.TestCase):

  def test_basic(self):

    def f(a):
      assert a, 'testmsg'
      return a

    tr = self.transform(f, (functions, asserts, return_statements))

    op = tr(constant_op.constant(False))
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError, 'testmsg'):
      self.evaluate(op)


if __name__ == '__main__':
  test.main()
