###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, July 14, 2025.                                              #
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
"""Tests for machina.python.framework._py_context_manager."""

from machina.python.framework import _py_context_manager
from machina.python.framework import test_util
from machina.python.platform import googletest


class TestContextManager(object):

  def __init__(self, behavior="basic"):
    self.log = []
    self.behavior = behavior

  def __enter__(self):
    self.log.append("__enter__()")
    if self.behavior == "raise_from_enter":
      raise ValueError("exception in __enter__")
    return "var"

  def __exit__(self, ex_type, ex_value, ex_tb):
    self.log.append("__exit__(%s, %s, %s)" % (ex_type, ex_value, ex_tb))
    if self.behavior == "raise_from_exit":
      raise ValueError("exception in __exit__")
    if self.behavior == "suppress_exception":
      return True


# Expected log when the body doesn't raise an exception.
NO_EXCEPTION_LOG = """\
__enter__()
body('var')
__exit__(None, None, None)"""

# Expected log when the body does raise an exception.  (Regular expression.)
EXCEPTION_LOG = """\
__enter__\\(\\)
body\\('var'\\)
__exit__\\(<class 'ValueError'>, Foo, <traceback object.*>\\)"""


class OpDefUtilTest(test_util.TensorFlowTestCase):

  def testBasic(self):
    cm = TestContextManager()

    def body(var):
      cm.log.append("body(%r)" % var)

    _py_context_manager.test_py_context_manager(cm, body)
    self.assertEqual("\n".join(cm.log), NO_EXCEPTION_LOG)

  def testBodyRaisesException(self):
    cm = TestContextManager()

    def body(var):
      cm.log.append("body(%r)" % var)
      raise ValueError("Foo")

    with self.assertRaisesRegex(ValueError, "Foo"):
      _py_context_manager.test_py_context_manager(cm, body)
    self.assertRegex("\n".join(cm.log), EXCEPTION_LOG)

  def testEnterRaisesException(self):
    cm = TestContextManager("raise_from_enter")

    def body(var):
      cm.log.append("body(%r)" % var)

    with self.assertRaisesRegex(ValueError, "exception in __enter__"):
      _py_context_manager.test_py_context_manager(cm, body)
    self.assertEqual("\n".join(cm.log), "__enter__()")

  # Test behavior in unsupported case where __exit__ raises an exception.
  def testExitRaisesException(self):
    cm = TestContextManager("raise_from_exit")

    def body(var):
      cm.log.append("body(%r)" % var)

    # Note: this does *not* raise an exception (but does log a warning):
    _py_context_manager.test_py_context_manager(cm, body)
    self.assertEqual("\n".join(cm.log), NO_EXCEPTION_LOG)

  # Test behavior in unsupported case where __exit__ suppresses exception.
  def testExitSuppressesException(self):
    cm = TestContextManager("suppress_exception")

    def body(var):
      cm.log.append("body(%r)" % var)
      raise ValueError("Foo")

    with self.assertRaisesRegex(
        ValueError, "machina::PyContextManager::Enter does not support "
        "context managers that suppress exception"):
      _py_context_manager.test_py_context_manager(cm, body)
    self.assertRegex("\n".join(cm.log), EXCEPTION_LOG)


if __name__ == "__main__":
  googletest.main()
