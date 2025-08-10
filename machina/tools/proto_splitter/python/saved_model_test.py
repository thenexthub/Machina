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
"""Tests for SavedModelSplitter."""


from google.protobuf import message
from machina.core.protobuf import saved_model_pb2
from machina.python.platform import test
from machina.tools.proto_splitter import constants
from machina.tools.proto_splitter.python import saved_model
from machina.tools.proto_splitter.python import test_util


class SavedModelSplitterTest(test.TestCase):

  def _assert_chunk_sizes(self, chunks, max_size):
    """Asserts that all chunk proto sizes are <= max_size."""
    for chunk in chunks:
      if isinstance(chunk, message.Message):
        self.assertLessEqual(chunk.ByteSize(), max_size)

  def test_split_saved_model(self):
    sizes = [100, 100, 1000, 100, 1000, 500, 100, 100, 100]
    fn1 = [100, 100, 100]
    fn2 = [100, 500]
    fn3 = [100]
    fn4 = [100, 100]

    max_size = 500
    constants.debug_set_max_size(max_size)

    graph_def = test_util.make_graph_def_with_constant_nodes(
        sizes, fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4
    )
    proto = saved_model_pb2.SavedModel()
    proto.meta_graphs.add().graph_def.CopyFrom(graph_def)

    splitter = saved_model.SavedModelSplitter(proto)
    chunks, _ = splitter.split()
    self._assert_chunk_sizes(chunks, max_size)


if __name__ == "__main__":
  test.main()
