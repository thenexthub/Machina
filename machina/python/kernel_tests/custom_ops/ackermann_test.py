###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Tuesday, March 25, 2025.                                            #
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
"""Tests for custom user ops."""
import os

from machina.python.framework import load_library
from machina.python.framework import test_util
from machina.python.platform import resource_loader
from machina.python.platform import test


class AckermannTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testBasic(self):
    library_filename = os.path.join(resource_loader.get_data_files_path(),
                                    'ackermann_op.so')
    ackermann = load_library.load_op_library(library_filename)

    with self.cached_session():
      self.assertEqual(ackermann.ackermann().eval(), b'A(m, 0) == A(m-1, 1)')


if __name__ == '__main__':
  test.main()
