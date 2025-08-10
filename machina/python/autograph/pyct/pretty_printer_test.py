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
"""Tests for pretty_printer module."""

import ast
import textwrap

from machina.python.autograph.pyct import pretty_printer
from machina.python.platform import test


class PrettyPrinterTest(test.TestCase):

  def test_unicode_bytes(self):
    source = textwrap.dedent('''
    def f():
      return b'b', u'u', 'depends_py2_py3'
    ''')
    node = ast.parse(source)
    self.assertIsNotNone(pretty_printer.fmt(node))

  def test_format(self):
    node = ast.FunctionDef(
        name='f',
        args=ast.arguments(
            args=[ast.Name(id='a', ctx=ast.Param())],
            vararg=None,
            kwarg=None,
            defaults=[]),
        body=[
            ast.Return(
                ast.BinOp(
                    op=ast.Add(),
                    left=ast.Name(id='a', ctx=ast.Load()),
                    right=ast.Num(1)))
        ],
        decorator_list=[],
        returns=None)
    # Just checking for functionality, the color control characters make it
    # difficult to inspect the result.
    self.assertIsNotNone(pretty_printer.fmt(node))


if __name__ == '__main__':
  test.main()
