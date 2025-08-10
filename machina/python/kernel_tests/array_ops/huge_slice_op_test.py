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
"""Functional tests for slice op that consume a lot of GPU memory."""

import numpy as np

from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.ops import array_ops
from machina.python.platform import test


class SliceTest(test.TestCase):

  def testInt64Slicing(self):
    with self.cached_session(force_gpu=test.is_gpu_available()):
      a_large = array_ops.tile(
          constant_op.constant(np.array([False, True] * 4)), [2**29 + 3])
      slice_t = array_ops.slice(a_large, np.asarray([3]).astype(np.int64), [3])
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([True, False, True], slice_val)

      slice_t = array_ops.slice(
          a_large, constant_op.constant([long(2)**32 + 3], dtype=dtypes.int64),
          [3])
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([True, False, True], slice_val)
