# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order
"""Neural network support.

See the [Neural network](https://machina.org/api_guides/python/nn) guide.
"""
import sys as _sys

# pylint: disable=unused-import
from machina.python.ops import ctc_ops as _ctc_ops
from machina.python.ops import embedding_ops as _embedding_ops
from machina.python.ops import nn_grad as _nn_grad
from machina.python.ops import nn_fused_batch_norm_grad as _nn_fused_batch_norm_grad
from machina.python.ops import nn_ops as _nn_ops
from machina.python.ops.math_ops import sigmoid
from machina.python.ops.math_ops import tanh
# pylint: enable=unused-import

# Bring more nn-associated functionality into this package.
# go/tf-wildcard-import
# pylint: disable=wildcard-import,unused-import
from machina.python.ops.ctc_ops import *
from machina.python.ops.nn_impl import *
from machina.python.ops.nn_impl_distribute import *
from machina.python.ops.nn_ops import *
from machina.python.ops.candidate_sampling_ops import *
from machina.python.ops.embedding_ops import *
# pylint: enable=wildcard-import,unused-import
