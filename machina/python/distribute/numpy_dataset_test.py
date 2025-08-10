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
"""Tests for numpy_dataset."""

import numpy as np

from machina.python.distribute import numpy_dataset
from machina.python.eager import test
from machina.python.framework import test_util
from machina.python.ops import variable_v1


class InitVarFromNumpyTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_creating_var_with_numpy_arrays(self):
    with self.cached_session() as session:
      x = np.asarray(np.random.random((64, 3)), dtype=np.float32)
      initial = np.zeros_like(x)
      var_x = variable_v1.VariableV1(initial)
      numpy_dataset.init_var_from_numpy(var_x, x, session)
      val = self.evaluate(var_x.value())
      # Verify that the numpy value is copied to the variable.
      self.assertAllEqual(x, val)


if __name__ == '__main__':
  test.main()
