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
"""Tests for the XLATestCase test fixture base class."""

from machina.compiler.tests import xla_test
from machina.python.platform import test


class XlaTestCaseTestCase(test.TestCase):

  def testManifestEmptyLineDoesNotCatchAll(self):
    manifest = """
testCaseOne
"""
    disabled_regex, _ = xla_test.parse_disabled_manifest(manifest)
    self.assertEqual(disabled_regex, "testCaseOne")

  def testManifestWholeLineCommentDoesNotCatchAll(self):
    manifest = """# I am a comment
testCaseOne
testCaseTwo
"""
    disabled_regex, _ = xla_test.parse_disabled_manifest(manifest)
    self.assertEqual(disabled_regex, "testCaseOne|testCaseTwo")


if __name__ == "__main__":
  test.main()
