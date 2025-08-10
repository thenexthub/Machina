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
"""Logical boolean operators: not, and, or."""

from machina.python.framework import tensor_util
from machina.python.ops import cond as tf_cond
from machina.python.ops import gen_math_ops


def not_(a):
  """Functional form of "not"."""
  if tensor_util.is_tf_type(a):
    return _tf_not(a)
  return _py_not(a)


def _tf_not(a):
  """Implementation of the "not_" operator for TensorFlow."""
  return gen_math_ops.logical_not(a)


def _py_not(a):
  """Default Python implementation of the "not_" operator."""
  return not a


def and_(a, b):
  """Functional form of "and". Uses lazy evaluation semantics."""
  a_val = a()
  if tensor_util.is_tf_type(a_val):
    return _tf_lazy_and(a_val, b)
  return _py_lazy_and(a_val, b)


def _tf_lazy_and(cond, b):
  """Lazy-eval equivalent of "and" for Tensors."""
  # TODO(mdan): Enforce cond is scalar here?
  return tf_cond.cond(cond, b, lambda: cond)


def _py_lazy_and(cond, b):
  """Lazy-eval equivalent of "and" in Python."""
  return cond and b()


def or_(a, b):
  """Functional form of "or". Uses lazy evaluation semantics."""
  a_val = a()
  if tensor_util.is_tf_type(a_val):
    return _tf_lazy_or(a_val, b)
  return _py_lazy_or(a_val, b)


def _tf_lazy_or(cond, b):
  """Lazy-eval equivalent of "or" for Tensors."""
  # TODO(mdan): Enforce cond is scalar here?
  return tf_cond.cond(cond, lambda: cond, b)


def _py_lazy_or(cond, b):
  """Lazy-eval equivalent of "or" in Python."""
  return cond or b()


def eq(a, b):
  """Functional form of "equal"."""
  if tensor_util.is_tf_type(a) or tensor_util.is_tf_type(b):
    return _tf_equal(a, b)
  return _py_equal(a, b)


def _tf_equal(a, b):
  """Overload of "equal" for Tensors."""
  return gen_math_ops.equal(a, b)


def _py_equal(a, b):
  """Overload of "equal" that falls back to Python's default implementation."""
  return a == b


def not_eq(a, b):
  """Functional form of "not-equal"."""
  return not_(eq(a, b))
