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
"""Helper functions to add documentation to type aliases."""

from typing import Dict

# Not useful for builtin `help()`. But these get passed to the
# doc generator so that the description is still displayed on the site.
_EXTRA_DOCS: Dict[int, str] = {}


def document(obj, doc):
  """Adds a docstring to typealias by overriding the `__doc__` attribute.

  Note: Overriding `__doc__` is only possible after python 3.7.

  Args:
    obj: Typealias object that needs to be documented.
    doc: Docstring of the typealias. It should follow the standard pystyle
      docstring rules.
  """
  try:
    obj.__doc__ = doc
  except AttributeError:
    _EXTRA_DOCS[id(obj)] = doc
