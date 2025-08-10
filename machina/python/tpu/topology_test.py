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

"""Tests for topology.py."""

from machina.python.platform import test
from machina.python.tpu import topology


class TopologyTest(test.TestCase):

  def testSerialization(self):
    """Tests if the class is able to generate serialized strings."""
    original_topology = topology.Topology(
        mesh_shape=[1, 1, 1, 2],
        device_coordinates=[[[0, 0, 0, 0], [0, 0, 0, 1]]],
    )
    serialized_str = original_topology.serialized()
    new_topology = topology.Topology(serialized=serialized_str)

    # Make sure the topology recovered from serialized str is same as the
    # original topology.
    self.assertAllEqual(
        original_topology.mesh_shape, new_topology.mesh_shape)
    self.assertAllEqual(
        original_topology.device_coordinates, new_topology.device_coordinates)

if __name__ == "__main__":
  test.main()
