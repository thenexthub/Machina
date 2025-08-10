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
###############################################################################=
"""Util for converting a Python object to a Trackable."""


from machina.python.eager.polymorphic_function import saved_model_utils
from machina.python.framework import dtypes
from machina.python.framework import tensor_util
from machina.python.ops import resource_variable_ops
from machina.python.trackable import base
from machina.python.trackable import data_structures


def convert_to_trackable(obj, parent=None):
  """Converts `obj` to `Trackable`."""
  if isinstance(obj, base.Trackable):
    return obj
  obj = data_structures.wrap_or_unwrap(obj)
  if (tensor_util.is_tf_type(obj) and
      obj.dtype not in (dtypes.variant, dtypes.resource) and
      not resource_variable_ops.is_resource_variable(obj)):
    return saved_model_utils.TrackableConstant(obj, parent)
  if not isinstance(obj, base.Trackable):
    raise ValueError(f"Cannot convert {obj} to Trackable.")
  return obj
