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
"""Command line tool version of the extract_object_files module.

This command line tool version takes the archive file path and the destination
directory path as the positional command line arguments.
"""

import sys
from typing import Sequence
from machina.lite.ios import extract_object_files


def main(argv: Sequence[str]) -> None:
  if len(argv) != 3:
    raise RuntimeError('Usage: {} <archive_file> <dest_dir>'.format(argv[0]))

  archive_path = argv[1]
  dest_dir = argv[2]
  with open(archive_path, 'rb') as archive_file:
    extract_object_files.extract_object_files(archive_file, dest_dir)


if __name__ == '__main__':
  main(sys.argv)
