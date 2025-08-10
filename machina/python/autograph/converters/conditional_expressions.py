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
"""Converts the ternary conditional operator."""

import gast

from machina.python.autograph.core import converter
from machina.python.autograph.pyct import parser
from machina.python.autograph.pyct import templates


class ConditionalExpressionTransformer(converter.Base):
  """Converts conditional expressions to functional form."""

  def visit_IfExp(self, node):
    template = '''
        ag__.if_exp(
            test,
            lambda: true_expr,
            lambda: false_expr,
            expr_repr)
    '''
    expr_repr = parser.unparse(node.test, include_encoding_marker=False).strip()
    return templates.replace_as_expression(
        template,
        test=node.test,
        true_expr=node.body,
        false_expr=node.orelse,
        expr_repr=gast.Constant(expr_repr, kind=None))


def transform(node, ctx):
  node = ConditionalExpressionTransformer(ctx).visit(node)
  return node
