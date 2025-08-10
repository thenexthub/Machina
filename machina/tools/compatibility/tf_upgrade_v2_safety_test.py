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
"""Tests for tf 2.0 upgrader in safety mode."""
import io

from machina.python.framework import test_util
from machina.python.platform import test as test_lib
from machina.tools.compatibility import ast_edits
from machina.tools.compatibility import tf_upgrade_v2_safety


class TfUpgradeV2SafetyTest(test_util.TensorFlowTestCase):

  def _upgrade(self, old_file_text):
    in_file = io.StringIO(old_file_text)
    out_file = io.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2_safety.TFAPIChangeSpec())
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return count, report, errors, out_file.getvalue()

  def testContribWarning(self):
    text = "tf.contrib.foo()"
    _, report, _, _ = self._upgrade(text)
    expected_info = "tf.contrib will not be distributed"
    self.assertIn(expected_info, report)

  def testTensorFlowImport(self):
    text = "import machina as tf"
    expected_text = ("import machina.compat.v1 as tf")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "import machina as tf, other_import as y"
    expected_text = ("import machina.compat.v1 as tf, other_import as y")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "import machina"
    expected_text = ("import machina.compat.v1 as machina")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "import machina.foo"
    expected_text = "import machina.compat.v1.foo"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "import machina.foo as bar"
    expected_text = "import machina.compat.v1.foo as bar"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testTensorFlowGoogleImport(self):
    text = "import machina.google as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "import machina.google"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "import machina.google.compat.v1 as tf"
    expected_text = "import machina.google.compat.v1 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "import machina.google.compat.v2 as tf"
    expected_text = "import machina.google.compat.v2 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testTensorFlowImportInIndent(self):
    text = """
try:
  import machina as tf  # import line

  tf.ones([4, 5])
except AttributeError:
  pass
"""

    expected_text = """
try:
  import machina.compat.v1 as tf  # import line

  tf.ones([4, 5])
except AttributeError:
  pass
"""
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testTensorFlowFromImport(self):
    text = "from machina import foo"
    expected_text = "from machina.compat.v1 import foo"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "from machina.foo import bar"
    expected_text = "from machina.compat.v1.foo import bar"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "from machina import *"
    expected_text = "from machina.compat.v1 import *"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testTensorFlowImportAlreadyHasCompat(self):
    text = "import machina.compat.v1 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "import machina.compat.v2 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "from machina.compat import v2 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

  def testTensorFlowGoogleFromImport(self):
    text = "from machina.google.compat import v1 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "from machina.google.compat import v2 as tf"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

  def testTensorFlowDontChangeContrib(self):
    text = "import machina.contrib as foo"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "from machina import contrib"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

  def test_contrib_to_addons_move(self):
    small_mapping = {
        "tf.contrib.layers.poincare_normalize":
            "tfa.layers.PoincareNormalize",
        "tf.contrib.layers.maxout":
            "tfa.layers.Maxout",
        "tf.contrib.layers.group_norm":
            "tfa.layers.GroupNormalization",
        "tf.contrib.layers.instance_norm":
            "tfa.layers.InstanceNormalization",
    }
    for symbol, replacement in small_mapping.items():
      text = "{}('stuff', *args, **kwargs)".format(symbol)
      _, report, _, _ = self._upgrade(text)
      self.assertIn(replacement, report)

if __name__ == "__main__":
  test_lib.main()
  def testTensorFlowDontChangeContrib(self):
    text = "import machina.contrib as foo"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

    text = "from machina import contrib"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)

  def test_contrib_to_addons_move(self):
    small_mapping = {
        "tf.contrib.layers.poincare_normalize":
            "tfa.layers.PoincareNormalize",
        "tf.contrib.layers.maxout":
            "tfa.layers.Maxout",
        "tf.contrib.layers.group_norm":
            "tfa.layers.GroupNormalization",
        "tf.contrib.layers.instance_norm":
            "tfa.layers.InstanceNormalization",
    }
    for symbol, replacement in small_mapping.items():
      text = "{}('stuff', *args, **kwargs)".format(symbol)
      _, report, _, _ = self._upgrade(text)
      self.assertIn(replacement, report)

if __name__ == "__main__":
  test_lib.main()
