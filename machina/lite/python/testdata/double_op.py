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
"""Double op is a user's defined op for testing purpose."""

from machina.lite.python.testdata import double_op_wrapper
from machina.python.framework import dtypes
from machina.python.framework import load_library
from machina.python.platform import resource_loader

_double_op = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_double_op.so'))


def double(input_tensor):
  """Double op applies element-wise double to input data."""
  if (input_tensor.dtype != dtypes.int32 and
      input_tensor.dtype != dtypes.float32):
    raise ValueError('Double op only accept int32 or float32 values.')
  return double_op_wrapper.double(input_tensor)
