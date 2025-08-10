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
"""Tests for TPU Embeddings mid level API on TPU."""
from absl.testing import parameterized

from machina.python.compat import v2_compat
from machina.python.platform import test
from machina.python.tpu.tests import tpu_embedding_v2_correctness_base_test


class TPUEmbeddingCorrectnessTest(
    tpu_embedding_v2_correctness_base_test.TPUEmbeddingCorrectnessBaseTest):

  @parameterized.parameters(
      ['sgd', 'adagrad', 'adam', 'ftrl', 'adagrad_momentum'])
  def test_embedding(self, optimizer_name):
    if optimizer_name != 'sgd':
      self.skip_if_oss()
    self._test_embedding(
        optimizer_name, training=False, sparse=False, is_high_dimensional=True)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
