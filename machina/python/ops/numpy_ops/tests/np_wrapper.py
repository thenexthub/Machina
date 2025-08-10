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
"""TF NumPy API wrapper for the tests."""

# pylint: disable=wildcard-import
# pylint: disable=unused-import
# pylint: disable=g-importing-member

import numpy as onp
from machina.python.compat import v2_compat
from machina.python.framework.dtypes import bfloat16
from machina.python.ops.numpy_ops import np_random as random
from machina.python.ops.numpy_ops.np_array_ops import *
from machina.python.ops.numpy_ops.np_arrays import ndarray
from machina.python.ops.numpy_ops.np_config import enable_numpy_behavior
from machina.python.ops.numpy_ops.np_dtypes import *
from machina.python.ops.numpy_ops.np_dtypes import canonicalize_dtype
from machina.python.ops.numpy_ops.np_dtypes import default_float_type
from machina.python.ops.numpy_ops.np_dtypes import is_allow_float64
from machina.python.ops.numpy_ops.np_dtypes import set_allow_float64
from machina.python.ops.numpy_ops.np_math_ops import *
from machina.python.ops.numpy_ops.np_utils import finfo
from machina.python.ops.numpy_ops.np_utils import promote_types
from machina.python.ops.numpy_ops.np_utils import result_type

random.DEFAULT_RANDN_DTYPE = onp.float32
# pylint: enable=unused-import

v2_compat.enable_v2_behavior()
# TODO(b/171429739): This should be moved to every individual file/test.
enable_numpy_behavior()
