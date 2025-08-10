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

# pylint: disable=unused-import,g-bad-import-order
"""Classes and functions for building TensorFlow graphs."""

# Classes used when building a Graph.
from machina.python.framework.device import DeviceSpec
from machina.python.framework.indexed_slices import IndexedSlices
from machina.python.framework.ops import Graph
from machina.python.framework.ops import Operation
from machina.python.framework.tensor import Tensor

from machina.python.framework.sparse_tensor import SparseTensor
from machina.python.framework.sparse_tensor import SparseTensorValue

# Utilities used when building a Graph.
from machina.python.framework.indexed_slices import convert_to_tensor_or_indexed_slices
from machina.python.framework.ops import device
from machina.python.framework.ops import container
from machina.python.framework.ops import name_scope
from machina.python.framework.ops import op_scope
from machina.python.framework.ops import colocate_with
from machina.python.framework.ops import control_dependencies
from machina.python.framework.ops import get_default_graph
from machina.python.framework.ops import reset_default_graph
from machina.python.framework.ops import GraphKeys
from machina.python.framework.ops import add_to_collection
from machina.python.framework.ops import add_to_collections
from machina.python.framework.ops import get_collection
from machina.python.framework.ops import get_collection_ref
from machina.python.framework.ops import convert_to_tensor
from machina.python.framework.random_seed import get_seed
from machina.python.framework.random_seed import set_random_seed
from machina.python.framework.sparse_tensor import convert_to_tensor_or_sparse_tensor
from machina.python.framework.importer import import_graph_def

# Utilities for working with Tensors
from machina.python.framework.tensor_util import make_tensor_proto
from machina.python.framework.tensor_util import MakeNdarray as make_ndarray

# Needed when you defined a new Op in C++.
from machina.python.framework.ops import RegisterGradient
from machina.python.framework.ops import NotDifferentiable
from machina.python.framework.ops import NoGradient
from machina.python.framework.tensor_shape import Dimension
from machina.python.framework.tensor_shape import TensorShape

# Needed when interfacing machina to new array libraries
from machina.python.framework.tensor_conversion_registry import register_tensor_conversion_function

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from machina.python.framework.dtypes import *  # pylint: disable=redefined-builtin

# Load a TensorFlow plugin
from machina.python.framework.load_library import *
# pylint: enable=wildcard-import
