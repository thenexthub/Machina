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
"""Tests for the checkpoint/util.py."""

from machina.python.checkpoint import graph_view
from machina.python.checkpoint import save_util_v1
from machina.python.eager import test
from machina.python.ops import variables
from machina.python.saved_model import registration
from machina.python.trackable import autotrackable
from machina.python.util import object_identity


class TrackableWithRegisteredSaver(autotrackable.AutoTrackable):
  pass


registration.register_checkpoint_saver(
    name="RegisteredSaver",
    predicate=lambda x: isinstance(x, TrackableWithRegisteredSaver),
    save_fn=lambda trackables, file_prefix: [],
    restore_fn=lambda trackables, merged_prefix: None)


class SerializationTest(test.TestCase):

  def test_serialize_gathered_objects(self):
    root = autotrackable.AutoTrackable()
    root.v = variables.Variable(1.0)
    root.registered = TrackableWithRegisteredSaver()
    named_saveable_objects, _, _, registered_savers = (
        save_util_v1.serialize_gathered_objects(
            graph_view.ObjectGraphView(root)))

    self.assertLen(named_saveable_objects, 1)
    self.assertIs(named_saveable_objects[0].op, root.v)
    self.assertDictEqual(
        {"Custom.RegisteredSaver": {"registered": root.registered}},
        registered_savers)

  def test_serialize_gathered_objects_with_map(self):
    root = autotrackable.AutoTrackable()
    root.v = variables.Variable(1.0)
    root.registered = TrackableWithRegisteredSaver()

    copy_of_registered = TrackableWithRegisteredSaver()
    copy_of_v = variables.Variable(1.0)
    object_map = object_identity.ObjectIdentityDictionary()
    object_map[root.registered] = copy_of_registered
    object_map[root.v] = copy_of_v

    named_saveable_objects, _, _, registered_savers = (
        save_util_v1.serialize_gathered_objects(
            graph_view.ObjectGraphView(root), object_map))

    self.assertLen(named_saveable_objects, 1)
    self.assertIsNot(named_saveable_objects[0].op, root.v)
    self.assertIs(named_saveable_objects[0].op, copy_of_v)

    ret_value = registered_savers["Custom.RegisteredSaver"]["registered"]
    self.assertIsNot(root.registered, ret_value)
    self.assertIs(copy_of_registered, ret_value)


if __name__ == "__main__":
  test.main()
