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
"""Tests for machina.python.util.fast_module_type."""

from machina.python.platform import test
from machina.python.util import fast_module_type
FastModuleType = fast_module_type.get_fast_module_type_class()


class ChildFastModule(FastModuleType):

  def _getattribute1(self, name):  # pylint: disable=unused-argument
    return 2

  def _getattribute2(self, name):  # pylint: disable=unused-argument
    raise AttributeError("Pass to getattr")

  def _getattr(self, name):  # pylint: disable=unused-argument
    return 3


class EarlyAttrAccessModule(FastModuleType):

  def __init__(self, name):
    self.some_attr = 1
    super().__init__(name)  # pytype: disable=too-many-function-args


class FastModuleTypeTest(test.TestCase):

  def testAttributeAccessBeforeSuperInitDoesNotCrash(self):
    # Tests that the attribute access before super().__init__() does not crash.
    module = EarlyAttrAccessModule("early_attr")
    self.assertEqual(1, module.some_attr)

  def testMissingModuleNameCallDoesNotCrash(self):
    with self.assertRaises(TypeError):
      ChildFastModule()

  def testBaseGetattribute(self):
    # Tests that the default attribute lookup works.
    module = ChildFastModule("test")
    module.foo = 1
    self.assertEqual(1, module.foo)

  def testGetattributeCallback(self):
    # Tests that functionality of __getattribute__ can be set as a callback.
    module = ChildFastModule("test")
    FastModuleType.set_getattribute_callback(module,
                                             ChildFastModule._getattribute1)
    self.assertEqual(2, module.foo)

  def testGetattrCallback(self):
    # Tests that functionality of __getattr__ can be set as a callback.
    module = ChildFastModule("test")
    FastModuleType.set_getattribute_callback(module,
                                             ChildFastModule._getattribute2)
    FastModuleType.set_getattr_callback(module, ChildFastModule._getattr)
    self.assertEqual(3, module.foo)

  def testFastdictApis(self):
    module = ChildFastModule("test")
    # At first "bar" does not exist in the module's attributes
    self.assertFalse(module._fastdict_key_in("bar"))
    with self.assertRaisesRegex(KeyError, "module has no attribute 'bar'"):
      module._fastdict_get("bar")

    module._fastdict_insert("bar", 1)
    # After _fastdict_insert() the attribute is added.
    self.assertTrue(module._fastdict_key_in("bar"))
    self.assertEqual(1, module.bar)


if __name__ == "__main__":
  test.main()
