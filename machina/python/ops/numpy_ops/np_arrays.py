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
"""ndarray class."""

# pylint: disable=g-direct-machina-import

from machina.python.framework import dtypes
from machina.python.framework import tensor
from machina.python.framework import tensor_conversion
from machina.python.ops.numpy_ops import np_dtypes


def convert_to_tensor(value, dtype=None, dtype_hint=None):
  """Wrapper over `tf.convert_to_tensor`.

  Args:
    value: value to convert
    dtype: (optional) the type we would like it to be converted to.
    dtype_hint: (optional) soft preference for the type we would like it to be
      converted to. `tf.convert_to_tensor` will attempt to convert value to this
      type first, but will not fail if conversion is not possible falling back
      to inferring the type instead.

  Returns:
    Value converted to tf.Tensor.
  """
  # A safer version of `tf.convert_to_tensor` to work around b/149876037.
  # TODO(wangpeng): Remove this function once the bug is fixed.
  if (dtype is None and isinstance(value, int) and
      value >= 2**63):
    dtype = dtypes.uint64
  elif dtype is None and dtype_hint is None and isinstance(value, float):
    dtype = np_dtypes.default_float_type()
  return tensor_conversion.convert_to_tensor_v2_with_dispatch(
      value, dtype=dtype, dtype_hint=dtype_hint)


ndarray = tensor.Tensor
