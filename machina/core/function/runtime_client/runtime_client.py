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
"""Low level TF runtime client."""

# TF oddity: this import loads TF-specific dynamic libraries.
from machina.python import pywrap_machina  # pylint:disable=g-bad-import-order,unused-import

from machina.core.framework import function_pb2
from machina.core.function.runtime_client import runtime_client_pybind

GlobalEagerContext = runtime_client_pybind.GlobalEagerContext
GlobalPythonEagerContext = runtime_client_pybind.GlobalPythonEagerContext


# TODO(mdan): Map without adapters once pybind11_protobuf available
class Runtime(runtime_client_pybind.Runtime):

  def GetFunctionProto(self, name: str) -> function_pb2.FunctionDef:
    return function_pb2.FunctionDef.FromString(
        self.GetFunctionProtoString(name))

  def CreateFunction(self, function_def: function_pb2.FunctionDef):
    self.CreateFunctionFromString(function_def.SerializeToString())
