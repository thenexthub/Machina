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
"""Converts assert statements to their corresponding TF calls."""

import gast

from machina.python.autograph.core import converter
from machina.python.autograph.pyct import templates


class AssertTransformer(converter.Base):
  """Transforms Assert nodes to Call so they can be handled as functions."""

  def visit_Assert(self, node):
    self.generic_visit(node)

    # Note: The lone tf.Assert call will be wrapped with control_dependencies
    # by side_effect_guards.
    template = """
      ag__.assert_stmt(test, lambda: msg)
    """

    if node.msg is None:
      return templates.replace(
          template,
          test=node.test,
          msg=gast.Constant('Assertion error', kind=None))
    elif isinstance(node.msg, gast.Constant):
      return templates.replace(template, test=node.test, msg=node.msg)
    else:
      raise NotImplementedError('can only convert string messages for now.')


def transform(node, ctx):
  node = AssertTransformer(ctx).visit(node)
  return node
