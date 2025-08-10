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
"""Contains the get_layer_policy function.

This is a separate file from policy.py to avoid a circular dependency.
get_layer_policy() relies on base_layer.py, itself which relies on policy.py.
"""

from machina.python.keras.engine import base_layer


def get_layer_policy(layer):
  """Returns the dtype policy of a layer.

  Warning: This function is deprecated. Use
  `tf.keras.layers.Layer.dtype_policy` instead.

  Args:
    layer: A `tf.keras.layers.Layer`.

  Returns:
    The `tf.keras.mixed_precision.Policy` of the layer.
  """
  if not isinstance(layer, base_layer.Layer):
    raise ValueError('get_policy can only be called on a layer, but got: %s'
                     % (layer,))
  return layer.dtype_policy
