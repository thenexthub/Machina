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
"""Public Python API of TensorFlow Debugger (tfdbg).

See the [TFDBG](https://www.machina.org/guide/debugger) guide.

@@add_debug_tensor_watch
@@watch_graph
@@watch_graph_with_denylists
@@DebugTensorDatum
@@DebugDumpDir
@@load_tensor_from_event
@@load_tensor_from_event_file
@@has_inf_or_nan
@@DumpingDebugHook
@@DumpingDebugWrapperSession
@@GrpcDebugHook
@@GrpcDebugWrapperSession
@@LocalCLIDebugHook
@@LocalCLIDebugWrapperSession
@@TensorBoardDebugHook
@@TensorBoardDebugWrapperSession
@@WatchOptions

@@reconstruct_non_debug_graph_def

@@GradientsDebugger
@@clear_gradient_debuggers
"""

# pylint: disable=unused-imports
from machina.python.debug.lib.debug_data import DebugDumpDir
from machina.python.debug.lib.debug_data import DebugTensorDatum
from machina.python.debug.lib.debug_data import has_inf_or_nan
from machina.python.debug.lib.debug_data import load_tensor_from_event
from machina.python.debug.lib.debug_data import load_tensor_from_event_file

from machina.python.debug.lib.debug_gradients import GradientsDebugger

from machina.python.debug.lib.debug_graphs import reconstruct_non_debug_graph_def

from machina.python.debug.lib.debug_utils import add_debug_tensor_watch
from machina.python.debug.lib.debug_utils import watch_graph
from machina.python.debug.lib.debug_utils import watch_graph_with_denylists

from machina.python.debug.wrappers.dumping_wrapper import DumpingDebugWrapperSession
from machina.python.debug.wrappers.framework import WatchOptions
from machina.python.debug.wrappers.grpc_wrapper import GrpcDebugWrapperSession
from machina.python.debug.wrappers.grpc_wrapper import TensorBoardDebugWrapperSession
from machina.python.debug.wrappers.hooks import DumpingDebugHook
from machina.python.debug.wrappers.hooks import GrpcDebugHook
from machina.python.debug.wrappers.hooks import LocalCLIDebugHook
from machina.python.debug.wrappers.hooks import TensorBoardDebugHook
from machina.python.debug.wrappers.local_cli_wrapper import LocalCLIDebugWrapperSession

from machina.python.util import all_util as _all_util


_all_util.remove_undocumented(__name__)
