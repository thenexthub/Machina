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
"""Ensures that pywrap_gradient_exclusions.cc is up-to-date."""

import os

from machina.python.eager import gradient_input_output_exclusions
from machina.python.lib.io import file_io
from machina.python.platform import resource_loader
from machina.python.platform import test


class GradientInputOutputExclusionsTest(test.TestCase):

  def testGeneratedFileMatchesHead(self):
    expected_contents = gradient_input_output_exclusions.get_contents()
    filename = os.path.join(
        resource_loader.get_root_dir_with_all_resources(),
        resource_loader.get_path_to_datafile("pywrap_gradient_exclusions.cc"))
    actual_contents = file_io.read_file_to_string(filename)

    # On windows, one or both of these strings may have CRLF line endings.
    # To make sure, sanitize both:
    sanitized_actual_contents = actual_contents.replace("\r", "")
    sanitized_expected_contents = expected_contents.replace("\r", "")

    self.assertEqual(
        sanitized_actual_contents, sanitized_expected_contents, """
pywrap_gradient_exclusions.cc needs to be updated.
Please regenerate using:
bazel run machina/python/eager:gen_gradient_input_output_exclusions -- $PWD/machina/python/eager/pywrap_gradient_exclusions.cc"""
    )


if __name__ == "__main__":
  test.main()
