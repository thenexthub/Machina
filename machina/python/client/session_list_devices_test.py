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

"""Tests for machina.python.client.session.Session's list_devices API."""

from machina.core.protobuf import cluster_pb2
from machina.core.protobuf import config_pb2
from machina.python.client import pywrap_tf_session as tf_session
from machina.python.client import session
from machina.python.framework import errors
from machina.python.framework import ops
from machina.python.framework import test_util
from machina.python.platform import googletest
from machina.python.training import server_lib


class SessionListDevicesTest(test_util.TensorFlowTestCase):

  def testListDevices(self):
    with session.Session() as sess:
      devices = sess.list_devices()
      self.assertTrue('/job:localhost/replica:0/task:0/device:CPU:0' in set(
          [d.name for d in devices]), devices)
      # All valid device incarnations must be non-zero.
      self.assertTrue(all(d.incarnation != 0 for d in devices))

  def testInvalidDeviceNumber(self):
    opts = tf_session.TF_NewSessionOptions()
    with ops.get_default_graph()._c_graph.get() as c_graph:
      c_session = tf_session.TF_NewSession(c_graph, opts)
    raw_device_list = tf_session.TF_SessionListDevices(c_session)
    size = tf_session.TF_DeviceListCount(raw_device_list)
    with self.assertRaises(errors.InvalidArgumentError):
      tf_session.TF_DeviceListMemoryBytes(raw_device_list, size)
    tf_session.TF_DeleteDeviceList(raw_device_list)
    tf_session.TF_CloseSession(c_session)

  def testListDevicesGrpcSession(self):
    server = server_lib.Server.create_local_server()
    with session.Session(server.target) as sess:
      devices = sess.list_devices()
      self.assertTrue(
          '/job:localhost/replica:0/task:0/device:CPU:0' in set(
              [d.name for d in devices]), devices)
      # All valid device incarnations must be non-zero.
      self.assertTrue(all(d.incarnation != 0 for d in devices))

  def testListDevicesClusterSpecPropagation(self):
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()

    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)
    with session.Session(server1.target, config=config) as sess:
      devices = sess.list_devices()
      device_names = set(d.name for d in devices)
      self.assertTrue(
          '/job:worker/replica:0/task:0/device:CPU:0' in device_names)
      self.assertTrue(
          '/job:worker/replica:0/task:1/device:CPU:0' in device_names)
      # All valid device incarnations must be non-zero.
      self.assertTrue(all(d.incarnation != 0 for d in devices))


if __name__ == '__main__':
  googletest.main()
