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
"""Keras injection tests."""

import machina as tf

from machina.python.eager import test


# Some of the Keras code load should be triggered so that it will inject proper
# functionality like registering the optimizer class for SavedModel.
class KerasInjectionTest(tf.test.TestCase):

  def test_keras_optimizer_injected(self):
    save_path = test.test_src_dir_path(
        'cc/saved_model/testdata/OptimizerSlotVariableModule')
    _ = tf.saved_model.load(save_path)
    # Make sure keras optimizers are registed without accessing keras code
    # when loading a model with optimizers
    self.assertIn(
        'optimizer', tf.__internal__.saved_model.load.registered_identifiers()
    )


if __name__ == '__main__':
  tf.test.main()
