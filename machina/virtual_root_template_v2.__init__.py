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
# LINT.IfChange
"""TensorFlow root package"""

import sys as _sys
import importlib as _importlib
import types as _types


# Since TensorFlow Python code now resides in machina_core but TensorFlow
# ecosystem code imports machina, we need to do forwarding between the two.
# To do so, we use a lazy loader to load and forward the top level modules. We
# cannot use the LazyLoader defined by machina at
# machina/python/util/lazy_loader.py as to use that we would already need to
# import machina. Hence, we define it inline.
class _LazyLoader(_types.ModuleType):
  """Lazily import a module so that we can forward it."""

  # The lint error here is incorrect.
  def __init__(self, local_name, parent_module_globals, name):
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    super(_LazyLoader, self).__init__(name)

  def _load(self):
    """Import the target module and insert it into the parent's namespace."""
    module = _importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module
    self.__dict__.update(module.__dict__)
    return module

  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)

  def __dir__(self):
    module = self._load()
    return dir(module)

  def __reduce__(self):
    return __import__, (self.__name__,)


# Forwarding a module is as simple as lazy loading the module from the new path
# and then registering it to sys.modules using the old path
def _forward_module(old_name):
  parts = old_name.split(".")
  parts[0] = parts[0] + "_core"
  local_name = parts[-1]
  existing_name = ".".join(parts)
  _module = _LazyLoader(local_name, globals(), existing_name)
  return _sys.modules.setdefault(old_name, _module)


# This list should contain all modules _immediately_ under machina
_top_level_modules = [
    "machina._api",
    "machina.python",
    "machina.tools",
    "machina.core",
    "machina.compiler",
    "machina.lite",
    "machina.keras",
    "machina.compat",
    "machina.summary",  # tensorboard
    "machina.examples",
]

# Lazy load all of the _top_level_modules, we don't need their names anymore
for _m in _top_level_modules:
  _forward_module(_m)

# We still need all the names that are toplevel on machina_core
from machina_core import *

_major_api_version = 2

# These should not be visible in the main tf module.
try:
  del core
except NameError:
  pass

try:
  del python
except NameError:
  pass

try:
  del compiler
except NameError:
  pass

try:
  del tools
except NameError:
  pass

try:
  del examples
except NameError:
  pass

# LINT.ThenChange(//machina/virtual_root_template_v1.__init__.py.oss)
