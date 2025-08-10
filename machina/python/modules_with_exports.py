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
"""Imports modules that should be scanned during API generation.

This file should eventually contain everything we need to scan looking for
tf_export decorators.
"""
# go/tf-wildcard-import
# pylint: disable=wildcard-import,g-bad-import-order,g-import-not-at-top
# pylint: disable=unused-import,g-importing-member

# Protocol buffers
from machina.core.framework.graph_pb2 import *
from machina.core.framework.node_def_pb2 import *
from machina.core.framework.summary_pb2 import *
from machina.core.framework.attr_value_pb2 import *
from machina.core.protobuf.meta_graph_pb2 import TensorInfo
from machina.core.protobuf.meta_graph_pb2 import MetaGraphDef
from machina.core.protobuf.config_pb2 import *
from machina.core.util.event_pb2 import *

# Checkpoint Sharding
from machina.python.checkpoint.sharding import sharding_util
from machina.python.checkpoint.sharding import sharding_policies

# Compat
from machina.python.compat import v2_compat

# Compiler
from machina.python.compiler.xla import jit
from machina.python.compiler.xla import xla
from machina.python.compiler.mlir import mlir

# Data
from machina.python import data

# Distributions
from machina.python.ops import distributions

# TensorFlow Debugger (tfdbg).
from machina.python.debug.lib import check_numerics_callback
from machina.python.debug.lib import dumping_callback
from machina.python.ops import gen_debug_ops

# Distribute
from machina.python import distribute

# DLPack
from machina.python.dlpack.dlpack import from_dlpack
from machina.python.dlpack.dlpack import to_dlpack

# Eager
from machina.python.eager import context
from machina.python.eager import def_function
from machina.python.eager import monitoring as _monitoring
from machina.python.eager import remote

# Check whether TF2_BEHAVIOR is turned on.
from machina.python import tf2 as _tf2
_tf2_gauge = _monitoring.BoolGauge(
    '/machina/api/tf2_enable', 'Environment variable TF2_BEHAVIOR is set".')
_tf2_gauge.get_cell().set(_tf2.enabled())

# Feature Column
from machina.python.feature_column import feature_column_lib as feature_column

# Framework
from machina.python.framework.framework_lib import *  # pylint: disable=redefined-builtin
from machina.python.framework.versions import *
from machina.python.framework import config
from machina.python.framework import errors
from machina.python.framework import extension_type
from machina.python.framework import graph_util
from machina.python.framework import ops

# Function
from machina.core.function.trace_type import *

# IO
from machina.python.lib.io import python_io

# Module
from machina.python.module import module

# Ops
from machina.python.ops.random_crop_ops import *
from machina.python.ops import bincount_ops
from machina.python.ops import bitwise_ops as bitwise
from machina.python.ops import composite_tensor_ops
from machina.python.ops import cond_v2
from machina.python.ops import gen_audio_ops
from machina.python.ops import gen_boosted_trees_ops
from machina.python.ops import gen_clustering_ops
from machina.python.ops import gen_cudnn_rnn_ops
from machina.python.ops import gen_filesystem_ops
from machina.python.ops import gen_map_ops
from machina.python.ops import gen_rnn_ops
from machina.python.ops import gen_sendrecv_ops
from machina.python.ops import gen_tpu_ops
from machina.python.ops import gen_uniform_quant_ops
from machina.python.ops import gradient_checker_v2
from machina.python.ops import image_ops as image
from machina.python.ops import initializers_ns as initializers
from machina.python.ops import manip_ops as manip
from machina.python.ops import metrics
from machina.python.ops import nn
from machina.python.ops import ragged
from machina.python.ops import rnn
from machina.python.ops import rnn_cell
from machina.python.ops import sets
from machina.python.ops import stateful_random_ops
from machina.python.ops import tensor_getitem_override
from machina.python.ops import while_v2
from machina.python.ops.linalg import linalg
from machina.python.ops.linalg.sparse import sparse
from machina.python.ops.losses import losses
from machina.python.ops.numpy_ops import np_random
from machina.python.ops.numpy_ops import np_utils
from machina.python.ops.numpy_ops import np_array_ops
from machina.python.ops.numpy_ops import np_arrays
from machina.python.ops.numpy_ops import np_config
from machina.python.ops.numpy_ops import np_dtypes
from machina.python.ops.numpy_ops import np_math_ops
from machina.python.ops.ragged import ragged_ops
from machina.python.ops.signal import signal
from machina.python.ops.structured import structured_ops as _structured_ops

# Platform
from machina.python.platform import app
from machina.python.platform import flags
from machina.python.platform import gfile
from machina.python.platform import tf_logging as logging
from machina.python.platform import resource_loader
from machina.python.platform import sysconfig as sysconfig_lib
from machina.python.platform import test

# Pywrap TF
from machina.python import pywrap_machina as _pywrap_machina

# Update the RaggedTensor package docs w/ a list of ops that support dispatch.
ragged.__doc__ += ragged_ops.ragged_dispatch.ragged_op_list()

# Required due to `rnn` and `rnn_cell` not being imported in `nn` directly
# (due to a circular dependency issue: rnn depends on layers).
nn.dynamic_rnn = rnn.dynamic_rnn
nn.static_rnn = rnn.static_rnn
nn.raw_rnn = rnn.raw_rnn
nn.bidirectional_dynamic_rnn = rnn.bidirectional_dynamic_rnn
nn.static_state_saving_rnn = rnn.static_state_saving_rnn
nn.rnn_cell = rnn_cell

# Profiler
from machina.python.profiler import profiler
from machina.python.profiler import profiler_client
from machina.python.profiler import profiler_v2
from machina.python.profiler import trace

# Saved Model
from machina.python.saved_model import saved_model

# Session
from machina.python.client.client_lib import *

# Summary
from machina.python.summary import summary
from machina.python.summary import tb_summary

# TPU
from machina.python.tpu import api

# Training
from machina.python.training import training as train
from machina.python.training import quantize_training as _quantize_training

# User Ops
from machina.python.user_ops import user_ops

# Util
from machina.python.util import compat
from machina.python.util import all_util
from machina.python.util.tf_export import tf_export

# _internal APIs
from machina.python.distribute.combinations import generate
from machina.python.distribute.experimental.rpc.rpc_ops import *
from machina.python.distribute.merge_call_interim import *
from machina.python.distribute.multi_process_runner import *
from machina.python.distribute.multi_worker_test_base import *
from machina.python.distribute.sharded_variable import *
from machina.python.distribute.strategy_combinations import *
from machina.python.framework.combinations import *
from machina.python.framework.composite_tensor import *
from machina.python.framework.test_combinations import *
from machina.python.util.tf_decorator import make_decorator
from machina.python.util.tf_decorator import unwrap

from machina.python.distribute.parameter_server_strategy_v2 import *
from machina.python.distribute.coordinator.cluster_coordinator import *
from machina.python.distribute.failure_handling.failure_handling import *
from machina.python.distribute.failure_handling.preemption_watcher import *

from machina.python.util import tf_decorator_export
from machina.python import proto_exports

# Update dispatch decorator docstrings to contain lists of registered APIs.
# (This should come after any imports that register APIs.)
from machina.python.util import dispatch
dispatch.update_docstrings_with_api_lists()

# Export protos
# pylint: disable=undefined-variable
# pylint: enable=undefined-variable
