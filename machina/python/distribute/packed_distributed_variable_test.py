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
from machina.python.distribute import device_util
from machina.python.distribute import packed_distributed_variable
from machina.python.eager import context
from machina.python.eager import def_function
from machina.python.framework import config
from machina.python.framework import ops
from machina.python.framework import test_util
from machina.python.ops import math_ops
from machina.python.ops import resource_variable_ops
from machina.python.platform import test
from machina.python.saved_model import load
from machina.python.saved_model import save
from machina.python.trackable import autotrackable


class TestExportArchive(autotrackable.AutoTrackable):

  def __init__(self):
    with ops.device('/cpu:0'):
      v0 = resource_variable_ops.ResourceVariable(1.0, name='var0')
    with ops.device('/cpu:1'):
      v1 = resource_variable_ops.ResourceVariable(2.0, name='var1')

    self._packed_var = packed_distributed_variable.PackedDistributedVariable(
        [v0, v1]
    )
    self._fn = def_function.function(self.update_var)

  @def_function.function
  def update_var(self):
    self._packed_var.assign_add(3.0).assign_sub(1.0)

  def save_function(self, directory):
    save.save(self, directory)


class PackedDistributedVariableTest(test.TestCase):

  def setUp(self):
    super(PackedDistributedVariableTest, self).setUp()
    cpus = config.list_physical_devices('CPU')
    # Set 2 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
    ])

  def testPackedVariable(self):
    with ops.device('/cpu:0'):
      v0 = resource_variable_ops.ResourceVariable(1.0, name='var0')
    with ops.device('/cpu:1'):
      v1 = resource_variable_ops.ResourceVariable(2.0, name='var1')

    packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])
    self.assertFalse(packed_var.handle.is_packed)
    self.assertTrue(packed_var.is_initialized)

    with ops.device('/cpu:0'):
      self.assertAllEqual(packed_var.get_var_on_current_device(), v0)
      val0 = packed_var.assign(2.0).assign_add(1.0)
      self.assertAllEqual(val0, 3.0)

    with ops.device('/cpu:1'):
      self.assertAllEqual(packed_var.get_var_on_current_device(), v1)
      val0 = packed_var.assign(2.0).assign_add(1.0)
      self.assertAllEqual(val0, 3.0)

    @def_function.function
    def update_var():
      self.assertTrue(packed_var.handle.is_packed)
      with ops.device('/cpu:0'):
        packed_var.assign_add(3.0).assign_sub(1.0)
        read0 = packed_var.value()
      with ops.device('/cpu:1'):
        packed_var.assign_sub(4.0).assign_sub(2.0)
        read1 = packed_var.value()

      return read0, read1

    self.assertAllEqual(update_var(), (5.0, -3.0))

  def testPackedVariableSaveAndRestore(self):
    with ops.device('/cpu:0'):
      v0 = resource_variable_ops.ResourceVariable(1.0, name='var0')
    with ops.device('/cpu:1'):
      v1 = resource_variable_ops.ResourceVariable(2.0, name='var1')

    packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])

    with ops.device('/cpu:0'):
      self.assertAllEqual(packed_var.get_var_on_current_device(), v0)
      val0 = packed_var.assign(2.0).assign_add(1.0)
      self.assertAllEqual(val0, 3.0)

    with ops.device('/cpu:1'):
      self.assertAllEqual(packed_var.get_var_on_current_device(), v1)
      val0 = packed_var.assign(2.0).assign_add(1.0)
      self.assertAllEqual(val0, 3.0)

    export_dir = self.get_temp_dir()
    save.save(packed_var, export_dir)

    restored_packed_var = load.load(export_dir)
    self.assertAllEqual(restored_packed_var, packed_var)

  def testPackedVariableExport(self):
    export_dir = self.get_temp_dir()
    export_archive = TestExportArchive()
    export_archive.save_function(export_dir)

    # restored_packed_var = load.load(export_dir)
    # logging.info(f'Restored var {restored_packed_var} ')

  def testPackedVarAndDevice(self):
    device0 = device_util.canonicalize('/cpu:0')
    device1 = device_util.canonicalize('/cpu:1')

    with ops.device(device0):
      v0 = resource_variable_ops.ResourceVariable(1.0)
    with ops.device(device1):
      v1 = resource_variable_ops.ResourceVariable(2.0)

    packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])

    packed_var0 = packed_distributed_variable.PackedVarAndDevice(
        packed_var, device0)
    self.assertFalse(packed_var0.handle.is_packed)
    self.assertAllEqual(math_ops.mul(packed_var0, 2.0), 2.0)

    packed_var1 = packed_distributed_variable.PackedVarAndDevice(
        packed_var, device1)
    self.assertAllEqual(packed_var1.assign(3.0), 3.0)

    @def_function.function
    def func():
      self.assertTrue(packed_var.handle.is_packed)
      var0 = packed_distributed_variable.PackedVarAndDevice(packed_var, device0)
      var0.assign_add(3.0)
      var1 = packed_distributed_variable.PackedVarAndDevice(packed_var, device1)
      return var0.value(), math_ops.add(var1, 2.0)

    self.assertAllEqual(func(), (4.0, 5.0))

  @test_util.assert_no_garbage_created
  def testNoGarbage(self):
    device0 = device_util.canonicalize('/cpu:0')
    device1 = device_util.canonicalize('/cpu:1')

    with ops.device(device0):
      v0 = resource_variable_ops.ResourceVariable(1.0)
    with ops.device(device1):
      v1 = resource_variable_ops.ResourceVariable(2.0)

    packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])
    # This needs a workaround to avoid creating reference cycles if the
    # attribute doesn't exist.
    hasattr(packed_var.on_device('/cpu:0'), 'nonexist')


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
