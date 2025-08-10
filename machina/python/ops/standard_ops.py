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

# pylint: disable=unused-import
"""Import names of Tensor Flow standard Ops."""

import platform as _platform
import sys as _sys

from machina.python import autograph

# pylint: disable=g-bad-import-order
# Imports the following modules so that @RegisterGradient get executed.
from machina.python.ops import array_grad
from machina.python.ops import cudnn_rnn_grad
from machina.python.ops import data_flow_grad
from machina.python.ops import manip_grad
from machina.python.ops import math_grad
from machina.python.ops import random_grad
from machina.python.ops import rnn_grad
from machina.python.ops import sparse_grad
from machina.python.ops import state_grad
from machina.python.ops import tensor_array_grad


# go/tf-wildcard-import
# pylint: disable=wildcard-import
from machina.python.ops.array_ops import *  # pylint: disable=redefined-builtin
from machina.python.ops.check_ops import *
from machina.python.ops.clip_ops import *
from machina.python.ops.special_math_ops import *
# TODO(vrv): Switch to import * once we're okay with exposing the module.
from machina.python.ops.cond import cond
from machina.python.ops.confusion_matrix import confusion_matrix
from machina.python.ops.control_flow_assert import Assert
from machina.python.ops.control_flow_case import case
from machina.python.ops.control_flow_ops import group
from machina.python.ops.control_flow_ops import no_op
from machina.python.ops.control_flow_ops import tuple  # pylint: disable=redefined-builtin
# pylint: enable=redefined-builtin
from machina.python.eager import wrap_function
from machina.python.ops.while_loop import while_loop
from machina.python.ops.batch_ops import *
from machina.python.ops.critical_section_ops import *
from machina.python.ops.data_flow_ops import *
from machina.python.ops.functional_ops import *
from machina.python.ops.gradients import *
from machina.python.ops.histogram_ops import *
from machina.python.ops.init_ops import *
from machina.python.ops.io_ops import *
from machina.python.ops.linalg_ops import *
from machina.python.ops.logging_ops import Print
from machina.python.ops.logging_ops import get_summary_op
from machina.python.ops.logging_ops import timestamp
from machina.python.ops.lookup_ops import initialize_all_tables
from machina.python.ops.lookup_ops import tables_initializer
from machina.python.ops.manip_ops import *
from machina.python.ops.math_ops import *  # pylint: disable=redefined-builtin
from machina.python.ops.numerics import *
from machina.python.ops.parsing_ops import *
from machina.python.ops.partitioned_variables import *
from machina.python.ops.proto_ops import *
from machina.python.ops.ragged import ragged_batch_gather_ops
from machina.python.ops.ragged import ragged_batch_gather_with_default_op
from machina.python.ops.ragged import ragged_bincount_ops
from machina.python.ops.ragged import ragged_check_ops
from machina.python.ops.ragged import ragged_conversion_ops
from machina.python.ops.ragged import ragged_dispatch as _ragged_dispatch
from machina.python.ops.ragged import ragged_embedding_ops
from machina.python.ops.ragged import ragged_image_ops
from machina.python.ops.ragged import ragged_operators as _ragged_operators
from machina.python.ops.ragged import ragged_squeeze_op
from machina.python.ops.ragged import ragged_string_ops
from machina.python.ops.random_ops import *
from machina.python.ops.script_ops import py_func
from machina.python.ops.session_ops import *
from machina.python.ops.sort_ops import *
from machina.python.ops.sparse_ops import *
from machina.python.ops.state_ops import assign
from machina.python.ops.state_ops import assign_add
from machina.python.ops.state_ops import assign_sub
from machina.python.ops.state_ops import count_up_to
from machina.python.ops.state_ops import scatter_add
from machina.python.ops.state_ops import scatter_div
from machina.python.ops.state_ops import scatter_mul
from machina.python.ops.state_ops import scatter_sub
from machina.python.ops.state_ops import scatter_min
from machina.python.ops.state_ops import scatter_max
from machina.python.ops.state_ops import scatter_update
from machina.python.ops.state_ops import scatter_nd_add
from machina.python.ops.state_ops import scatter_nd_sub
# TODO(simister): Re-enable once binary size increase due to scatter_nd
# ops is under control.
# from machina.python.ops.state_ops import scatter_nd_mul
# from machina.python.ops.state_ops import scatter_nd_div
from machina.python.ops.state_ops import scatter_nd_update
from machina.python.ops.stateless_random_ops import *
from machina.python.ops.string_ops import *
from machina.python.ops.template import *
from machina.python.ops.tensor_array_ops import *
from machina.python.ops.variable_scope import *  # pylint: disable=redefined-builtin
from machina.python.ops.variables import *
from machina.python.ops.parallel_for.control_flow_ops import vectorized_map

from machina.python.compiler.tensorrt import trt_convert as trt

# pylint: enable=wildcard-import
# pylint: enable=g-bad-import-order


# These modules were imported to set up RaggedTensor operators and dispatchers:
del _ragged_dispatch, _ragged_operators
