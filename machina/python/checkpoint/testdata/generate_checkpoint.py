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
###############################################################################=
"""Standalone utility to generate some test saved models."""

from absl import app

from machina.python.checkpoint import checkpoint
from machina.python.compat import v2_compat
from machina.python.framework import dtypes
from machina.python.module import module
from machina.python.ops import lookup_ops
from machina.python.ops import variables


class TableModule(module.Module):
  """Three vars (one in a sub-module) and compute method."""

  def __init__(self):
    default_value = -1
    empty_key = 0
    deleted_key = -1
    self.lookup_table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        initial_num_buckets=32)
    self.lookup_table.insert(1, 1)
    self.lookup_table.insert(2, 4)


class VariableModule(module.Module):

  def __init__(self):
    self.v = variables.Variable([1., 2., 3.])
    self.w = variables.Variable([4., 5.])

MODULE_CTORS = {
    "TableModule": TableModule,
    "VariableModule": VariableModule,
}


def main(args):
  if len(args) != 3:
    print("Expected: {export_path} {ModuleName}")
    print("Allowed ModuleNames:", MODULE_CTORS.keys())
    return 1

  _, export_path, module_name = args
  module_ctor = MODULE_CTORS.get(module_name)
  if not module_ctor:
    print("Expected ModuleName to be one of:", MODULE_CTORS.keys())
    return 2

  tf_module = module_ctor()
  ckpt = checkpoint.Checkpoint(tf_module)
  ckpt.write(export_path)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  app.run(main)
