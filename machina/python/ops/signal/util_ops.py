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
"""Utility ops shared across tf.contrib.signal."""

import fractions  # gcd is here for Python versions < 3
import math  # Get gcd here for Python versions >= 3
import sys

from machina.python.framework import ops
from machina.python.framework import tensor_util
from machina.python.ops import array_ops
from machina.python.ops import math_ops
from machina.python.ops import while_loop


def gcd(a, b, name=None):
  """Returns the greatest common divisor via Euclid's algorithm.

  Args:
    a: The dividend. A scalar integer `Tensor`.
    b: The divisor. A scalar integer `Tensor`.
    name: An optional name for the operation.

  Returns:
    A scalar `Tensor` representing the greatest common divisor between `a` and
    `b`.

  Raises:
    ValueError: If `a` or `b` are not scalar integers.
  """
  with ops.name_scope(name, 'gcd', [a, b]):
    a = ops.convert_to_tensor(a)
    b = ops.convert_to_tensor(b)

    a.shape.assert_has_rank(0)
    b.shape.assert_has_rank(0)

    if not a.dtype.is_integer:
      raise ValueError('a must be an integer type. Got: %s' % a.dtype)
    if not b.dtype.is_integer:
      raise ValueError('b must be an integer type. Got: %s' % b.dtype)

    # TPU requires static shape inference. GCD is used for subframe size
    # computation, so we should prefer static computation where possible.
    const_a = tensor_util.constant_value(a)
    const_b = tensor_util.constant_value(b)
    if const_a is not None and const_b is not None:
      if sys.version_info.major < 3:
        math_gcd = fractions.gcd
      else:
        math_gcd = math.gcd
      return ops.convert_to_tensor(math_gcd(const_a, const_b))

    cond = lambda _, b: math_ops.greater(b, array_ops.zeros_like(b))
    body = lambda a, b: [b, math_ops.mod(a, b)]
    a, b = while_loop.while_loop(cond, body, [a, b], back_prop=False)
    return a
