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
"""Test cases for debug XLA dumps."""

import glob
import os

import numpy as np

from machina.compiler.tests import xla_test
from machina.python.ops import array_ops
from machina.python.ops import math_ops
from machina.python.platform import googletest


class XlaDumpToDirTest(xla_test.XLATestCase):
  """Test that ensures --MACHINA_XLAFLAGS=--dump_to_xla=<dir> produces output."""

  def _compute(self):
    with self.session() as sess, self.device_scope():
      data = np.array([0], dtype=np.float32)
      indices = np.array([0], dtype=np.int32)
      d = array_ops.placeholder(data.dtype, shape=data.shape)
      i = array_ops.placeholder(indices.dtype, shape=indices.shape)
      sess.run(math_ops.segment_max_v2(data, indices, 1), {d: data, i: indices})

  def testDumpToTempDir(self):
    tmp_dir = self.create_tempdir().full_path
    os.environ['MACHINA_XLAFLAGS'] = '--xla_dump_to=' + tmp_dir
    self._compute()
    self.assertNotEmpty(glob.glob(os.path.join(tmp_dir, 'module_0*')))


if __name__ == '__main__':
  googletest.main()
