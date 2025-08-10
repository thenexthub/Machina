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
"""Common constants used by proto splitter modules."""

# The splitter algorithm isn't extremely precise, so the max is set to a little
# less than 2GB.
#
# TODO: b/380463192 - Consider fixing the split algorithm to handle edge cases
# accurately and raising the max size to 2GB.
_MAX_SIZE = 1 << 30


def debug_set_max_size(value: int) -> None:
  """Sets the max size allowed for each proto chunk (used for debugging only).

  Args:
    value: int byte size
  """
  global _MAX_SIZE
  _MAX_SIZE = value


def max_size() -> int:
  """Returns the maximum size each proto chunk."""
  return _MAX_SIZE
