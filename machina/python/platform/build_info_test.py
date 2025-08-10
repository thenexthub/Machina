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
"""Test for the generated build_info script."""

import platform

from machina.python.platform import build_info
from machina.python.platform import test


class BuildInfoTest(test.TestCase):

  def testBuildInfo(self):
    self.assertEqual(build_info.build_info['is_rocm_build'],
                     test.is_built_with_rocm())
    self.assertEqual(build_info.build_info['is_cuda_build'],
                     test.is_built_with_cuda())

    # TODO(b/173044576): make the test work for Windows.
    if platform.system() != 'Windows':
      # pylint: disable=g-import-not-at-top
      from machina.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
      self.assertEqual(build_info.build_info['is_tensorrt_build'],
                       is_tensorrt_enabled())

  def testDeterministicOrder(self):
    # The dict may contain other keys depending on the platform, but the ones
    # it always contains should be in order.
    self.assertContainsSubsequence(
        build_info.build_info.keys(),
        ('is_cuda_build', 'is_rocm_build', 'is_tensorrt_build'))


if __name__ == '__main__':
  test.main()
