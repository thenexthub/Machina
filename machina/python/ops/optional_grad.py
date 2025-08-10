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
"""Gradient functions for optional ops."""

from machina.python.framework import ops
from machina.python.ops import gen_optional_ops


@ops.RegisterGradient("OptionalFromValue")
def _OptionalFromValueGrad(op, grad):
  return gen_optional_ops.optional_get_value(
      grad, [t.dtype for t in op.inputs], [t.shape for t in op.inputs]
  )


@ops.RegisterGradient("OptionalGetValue")
def _OptionalGetValueGrad(unused_op, *grads):
  return gen_optional_ops.optional_from_value(grads)
