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
"""Tests for graph_only_ops."""

import numpy as np

from machina.python.eager import graph_only_ops
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.framework import test_util
from machina.python.ops import math_ops
from machina.python.platform import test


class GraphOnlyOpsTest(test_util.TensorFlowTestCase):

  def testGraphPlaceholder(self):
    with ops.Graph().as_default():
      x_tf = graph_only_ops.graph_placeholder(dtypes.int32, shape=(1,))
      y_tf = math_ops.square(x_tf)
      with self.cached_session() as sess:
        x = np.array([42])
        y = sess.run(y_tf, feed_dict={x_tf: np.array([42])})
        self.assertAllClose(np.square(x), y)


if __name__ == '__main__':
  test.main()
