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
"""A cache for FileWriters."""

import threading

from machina.python.framework import ops
from machina.python.summary.writer.writer import FileWriter
from machina.python.util.tf_export import tf_export


@tf_export(v1=['summary.FileWriterCache'])
class FileWriterCache(object):
  """Cache for file writers.

  This class caches file writers, one per directory.
  """
  # Cache, keyed by directory.
  _cache = {}

  # Lock protecting _FILE_WRITERS.
  _lock = threading.RLock()

  @staticmethod
  def clear():
    """Clear cached summary writers. Currently only used for unit tests."""
    with FileWriterCache._lock:
      # Make sure all the writers are closed now (otherwise open file handles
      # may hang around, blocking deletions on Windows).
      for item in FileWriterCache._cache.values():
        item.close()
      FileWriterCache._cache = {}

  @staticmethod
  def get(logdir):
    """Returns the FileWriter for the specified directory.

    Args:
      logdir: str, name of the directory.

    Returns:
      A `FileWriter`.
    """
    with FileWriterCache._lock:
      if logdir not in FileWriterCache._cache:
        FileWriterCache._cache[logdir] = FileWriter(
            logdir, graph=ops.get_default_graph())
      return FileWriterCache._cache[logdir]
