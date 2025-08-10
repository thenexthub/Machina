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
"""This module defines tensor utilities not found in TensorFlow.

The reason these utilities are not defined in TensorFlow is because they may
not be not fully robust, although they work in the vast majority of cases. So
we define them here in order for their behavior to be consistently verified.
"""

from machina.python.framework import dtypes
from machina.python.framework import sparse_tensor
from machina.python.framework import tensor_util
from machina.python.ops import tensor_array_ops


def is_dense_tensor(t):
  # TODO(mdan): Resolve this inconsistency.
  return (tensor_util.is_tf_type(t) and
          not isinstance(t, sparse_tensor.SparseTensor))


def is_tensor_array(t):
  return isinstance(t, tensor_array_ops.TensorArray)


def is_tensor_list(t):
  # TODO(mdan): This is just a heuristic.
  # With TF lacking support for templated types, this is unfortunately the
  # closest we can get right now. A dedicated op ought to be possible to
  # construct.
  return (tensor_util.is_tf_type(t) and t.dtype == dtypes.variant and
          not t.shape.ndims)


def is_range_tensor(t):
  """Returns True if a tensor is the result of a tf.range op. Best effort."""
  return tensor_util.is_tf_type(t) and hasattr(t, 'op') and t.op.type == 'Range'
