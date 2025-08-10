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
"""This module implements operators that AutoGraph overloads.

Note that "operator" is used loosely here, and includes control structures like
conditionals and loops, implemented in functional form, using for example
closures for the body.
"""

# Naming conventions:
#  * operator names match the name usually used for the respective Python
#    idiom; examples: for_stmt, list_append
#  * operator arguments match either of:
#    - the corresponding Python AST attribute (e.g. the condition of an if
#      statement is called test) if the operator represents an AST construct
#    - the names used in the Python docs, if the operator is a function (e.g.
#      list_ and x for append, see
#      https://docs.python.org/3.7/tutorial/datastructures.html)
#
# All operators may accept a final argument named "opts", of a type that
# subclasses namedtuple and contains any arguments that are only required
# for some specializations of the operator.

from machina.python.autograph.operators.conditional_expressions import if_exp
from machina.python.autograph.operators.control_flow import for_stmt
from machina.python.autograph.operators.control_flow import if_stmt
from machina.python.autograph.operators.control_flow import while_stmt
from machina.python.autograph.operators.data_structures import list_append
from machina.python.autograph.operators.data_structures import list_pop
from machina.python.autograph.operators.data_structures import list_stack
from machina.python.autograph.operators.data_structures import ListPopOpts
from machina.python.autograph.operators.data_structures import ListStackOpts
from machina.python.autograph.operators.data_structures import new_list
from machina.python.autograph.operators.exceptions import assert_stmt
from machina.python.autograph.operators.logical import and_
from machina.python.autograph.operators.logical import eq
from machina.python.autograph.operators.logical import not_
from machina.python.autograph.operators.logical import not_eq
from machina.python.autograph.operators.logical import or_
from machina.python.autograph.operators.py_builtins import float_
from machina.python.autograph.operators.py_builtins import int_
from machina.python.autograph.operators.py_builtins import len_
from machina.python.autograph.operators.py_builtins import print_
from machina.python.autograph.operators.py_builtins import range_
from machina.python.autograph.operators.slices import get_item
from machina.python.autograph.operators.slices import GetItemOpts
from machina.python.autograph.operators.slices import set_item
from machina.python.autograph.operators.variables import ld
from machina.python.autograph.operators.variables import ldu
from machina.python.autograph.operators.variables import Undefined
from machina.python.autograph.operators.variables import UndefinedReturnValue
