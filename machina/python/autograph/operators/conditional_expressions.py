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
"""Conditional expressions (e.g. the ternary if statement)."""


from machina.python.autograph.operators import control_flow
from machina.python.autograph.utils import tensors
from machina.python.ops import cond as tf_cond


def if_exp(cond, if_true, if_false, expr_repr):
  if tensors.is_dense_tensor(cond):
    return _tf_if_exp(cond, if_true, if_false, expr_repr)
  else:
    return _py_if_exp(cond, if_true, if_false)


def _tf_if_exp(cond, if_true, if_false, expr_repr):
  """Overload of if_exp that stages a TF cond."""
  # TODO(mdan): Use nonlocal once we no longer need to support py2.
  true_val = []
  false_val = []

  def true_fn():
    true_val.append(if_true())
    if true_val and false_val:
      control_flow.verify_single_cond_var(expr_repr, true_val[0], false_val[0])
    return true_val[0]

  def false_fn():
    false_val.append(if_false())
    if true_val and false_val:
      control_flow.verify_single_cond_var(expr_repr, true_val[0], false_val[0])
    return false_val[0]

  return tf_cond.cond(cond, true_fn, false_fn)


def _py_if_exp(cond, if_true, if_false):
  return if_true() if cond else if_false()
