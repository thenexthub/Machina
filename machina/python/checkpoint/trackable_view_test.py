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
"""Tests for the trackable view."""

from machina.python.checkpoint import trackable_view
from machina.python.eager import test
from machina.python.trackable import base


class TrackableViewTest(test.TestCase):

  def test_children(self):
    root = base.Trackable()
    leaf = base.Trackable()
    root._track_trackable(leaf, name="leaf")
    (current_name,
     current_dependency), = trackable_view.TrackableView.children(root).items()
    self.assertIs(leaf, current_dependency)
    self.assertEqual("leaf", current_name)

  def test_descendants(self):
    root = base.Trackable()
    leaf = base.Trackable()
    root._track_trackable(leaf, name="leaf")
    descendants = trackable_view.TrackableView(root).descendants()
    self.assertIs(2, len(descendants))
    self.assertIs(root, descendants[0])
    self.assertIs(leaf, descendants[1])


if __name__ == "__main__":
  test.main()
