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
###############################################################################=
"""Tests for loading SavedModels with optimizers."""


from machina.python.eager import test
from machina.python.ops import variables
from machina.python.saved_model import load


class LoadOptimizerTest(test.TestCase):

  def test_load_optimizer_without_keras(self):
    # Make sure that a SavedModel w/ optimizer can be loaded without the Keras
    # module imported.
    save_path = test.test_src_dir_path(
        "cc/saved_model/testdata/OptimizerSlotVariableModule")
    loaded = load.load(save_path)
    self.assertIsInstance(
        loaded.opt.get_slot(loaded.v, "v"), variables.Variable)


if __name__ == "__main__":
  test.main()
