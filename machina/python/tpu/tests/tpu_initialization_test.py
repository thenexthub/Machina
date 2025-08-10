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
"""Tests for TPU Initialization."""

from absl.testing import parameterized

from machina.python.compat import v2_compat
from machina.python.distribute.cluster_resolver import tpu_cluster_resolver
from machina.python.platform import test


class TPUInitializationTest(parameterized.TestCase, test.TestCase):

  def test_tpu_initialization(self):
    resolver = tpu_cluster_resolver.TPUClusterResolver('')
    tpu_cluster_resolver.initialize_tpu_system(resolver)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
