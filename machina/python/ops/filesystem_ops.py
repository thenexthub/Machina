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
"""Filesystem related operations."""

from machina.python.ops import gen_filesystem_ops as _gen_filesystem_ops


# pylint: disable=protected-access
def filesystem_set_configuration(scheme, key, value, name=None):
  """Set configuration of the file system.

  Args:
    scheme: File system scheme.
    key: The name of the configuration option.
    value: The value of the configuration option.
    name: A name for the operation (optional).

  Returns:
    None.
  """

  return _gen_filesystem_ops.file_system_set_configuration(
      scheme, key=key, value=value, name=name)


# pylint: enable=protected-access
