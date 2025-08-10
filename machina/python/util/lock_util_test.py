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
"""Tests for lock_util."""

import random
import time

from absl.testing import parameterized

from machina.python.platform import test
from machina.python.util import lock_util


class GroupLockTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 5, 10)
  def testGroups(self, num_groups):
    lock = lock_util.GroupLock(num_groups)
    num_threads = 10
    finished = set()

    def thread_fn(thread_id):
      time.sleep(random.random() * 0.1)
      group_id = thread_id % num_groups
      with lock.group(group_id):
        time.sleep(random.random() * 0.1)
        self.assertGreater(lock._group_member_counts[group_id], 0)
        for g, c in enumerate(lock._group_member_counts):
          if g != group_id:
            self.assertEqual(0, c)
        finished.add(thread_id)

    threads = [
        self.checkedThread(target=thread_fn, args=(i,))
        for i in range(num_threads)
    ]

    for i in range(num_threads):
      threads[i].start()
    for i in range(num_threads):
      threads[i].join()

    self.assertEqual(set(range(num_threads)), finished)


if __name__ == "__main__":
  test.main()
