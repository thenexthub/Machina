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
"""Tests for `multi_process_runner` for non-initialization."""

from machina.python.distribute import multi_process_runner
from machina.python.distribute import multi_worker_test_base
from machina.python.eager import test


class MultiProcessRunnerNoInitTest(test.TestCase):

  def test_not_calling_correct_main(self):

    def simple_func():
      return 'foobar'

    with self.assertRaisesRegex(multi_process_runner.NotInitializedError,
                                '`multi_process_runner` is not initialized.'):
      multi_process_runner.run(
          simple_func,
          multi_worker_test_base.create_cluster_spec(num_workers=1))


if __name__ == '__main__':
  # Intentionally not using `multi_process_runner.test_main()` so the error
  # would occur.
  test.main()
