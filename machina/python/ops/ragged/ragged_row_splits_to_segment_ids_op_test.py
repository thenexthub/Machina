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
"""Tests for the segment_id_ops.row_splits_to_segment_ids() op."""

from machina.python.framework import constant_op
from machina.python.framework import test_util
from machina.python.ops.ragged import segment_id_ops
from machina.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedSplitsToSegmentIdsOpTest(test_util.TensorFlowTestCase):

  def testDocStringExample(self):
    splits = [0, 3, 3, 5, 6, 9]
    expected = [0, 0, 0, 2, 2, 3, 4, 4, 4]
    segment_ids = segment_id_ops.row_splits_to_segment_ids(splits)
    self.assertAllEqual(segment_ids, expected)

  def testEmptySplits(self):
    # Note: the splits for an empty ragged tensor contains a single zero.
    segment_ids = segment_id_ops.row_splits_to_segment_ids([0])
    self.assertAllEqual(segment_ids, [])

  def testErrors(self):
    self.assertRaisesRegex(ValueError, r'Invalid row_splits: \[\]',
                           segment_id_ops.row_splits_to_segment_ids, [])
    self.assertRaisesRegex(ValueError, r'splits must have dtype int32 or int64',
                           segment_id_ops.row_splits_to_segment_ids,
                           constant_op.constant([0.5]))
    self.assertRaisesRegex(ValueError, r'Shape \(\) must have rank 1',
                           segment_id_ops.row_splits_to_segment_ids, 0)
    self.assertRaisesRegex(ValueError, r'Shape \(1, 1\) must have rank 1',
                           segment_id_ops.row_splits_to_segment_ids, [[0]])


if __name__ == '__main__':
  googletest.main()
