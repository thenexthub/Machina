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
"""Various context managers."""

import contextlib

from machina.python.framework import ops
from machina.python.ops import tensor_array_ops


def control_dependency_on_returns(return_value):
  """Create a TF control dependency on the return values of a function.

  If the function had no return value, a no-op context is returned.

  Args:
    return_value: The return value to set as control dependency.

  Returns:
    A context manager.
  """
  def control_dependency_handle(t):
    if isinstance(t, tensor_array_ops.TensorArray):
      return t.flow
    return t

  if return_value is None:
    return contextlib.contextmanager(lambda: (yield))()
  # TODO(mdan): Filter to tensor objects.
  if not isinstance(return_value, (list, tuple)):
    return_value = (return_value,)
  return_value = tuple(control_dependency_handle(t) for t in return_value)
  return ops.control_dependencies(return_value)
