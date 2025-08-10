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
"""Utilities for serializing Python objects."""

import numpy as np
import wrapt

from machina.python.framework import dtypes
from machina.python.framework import tensor_shape
from machina.python.util.compat import collections_abc


def get_json_type(obj):
  """Serializes any object to a JSON-serializable structure.

  Args:
      obj: the object to serialize

  Returns:
      JSON-serializable structure representing `obj`.

  Raises:
      TypeError: if `obj` cannot be serialized.
  """
  # if obj is a serializable Keras class instance
  # e.g. optimizer, layer
  if hasattr(obj, 'get_config'):
    return {'class_name': obj.__class__.__name__, 'config': obj.get_config()}

  # if obj is any numpy type
  if type(obj).__module__ == np.__name__:
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return obj.item()

  # misc functions (e.g. loss function)
  if callable(obj):
    return obj.__name__

  # if obj is a python 'type'
  if type(obj).__name__ == type.__name__:
    return obj.__name__

  if isinstance(obj, tensor_shape.Dimension):
    return obj.value

  if isinstance(obj, tensor_shape.TensorShape):
    return obj.as_list()

  if isinstance(obj, dtypes.DType):
    return obj.name

  if isinstance(obj, collections_abc.Mapping):
    return dict(obj)

  if obj is Ellipsis:
    return {'class_name': '__ellipsis__'}

  if isinstance(obj, wrapt.ObjectProxy):
    return obj.__wrapped__

  raise TypeError(f'Object {obj} is not JSON-serializable. You may implement '
                  'a `get_config()` method on the class '
                  '(returning a JSON-serializable dictionary) to make it '
                  'serializable.')
