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
"""Protobuf related tests."""

import numpy as np

from machina.python.framework import constant_op
from machina.python.framework import ops
from machina.python.platform import test


class ProtoTest(test.TestCase):

  # TODO(vrv): re-enable this test once we figure out how this can
  # pass the pip install test (where the user is expected to have
  # protobuf installed).
  def _testLargeProto(self):
    # create a constant of size > 64MB.
    a = constant_op.constant(np.zeros([1024, 1024, 17]))
    # Serialize the resulting graph def.
    gdef = a.op.graph.as_graph_def()
    serialized = gdef.SerializeToString()
    unserialized = ops.Graph().as_graph_def()
    # Deserialize back. Protobuf python library should support
    # protos larger than 64MB.
    unserialized.ParseFromString(serialized)
    self.assertProtoEquals(unserialized, gdef)


if __name__ == "__main__":
  test.main()
