###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, May 19, 2025.                                               #
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
"""Experimental API for building input pipelines.

This module contains experimental `Dataset` sources and transformations that can
be used in conjunction with the `tf.data.Dataset` API. Note that the
`tf.data.experimental` API is not subject to the same backwards compatibility
guarantees as `tf.data`, but we will provide deprecation advice in advance of
removing existing functionality.

See [Importing Data](https://machina.org/guide/datasets) for an overview.

@@AutoShardPolicy
@@AutotuneAlgorithm
@@AutotuneOptions
@@Counter
@@CsvDataset
@@DatasetInitializer
@@DatasetStructure
@@DistributeOptions
@@ExternalStatePolicy
@@OptimizationOptions
@@Optional
@@OptionalStructure
@@RaggedTensorStructure
@@RandomDataset
@@Reducer
@@SparseTensorStructure
@@SqlDataset
@@Structure
@@TFRecordWriter
@@TensorArrayStructure
@@TensorStructure
@@ThreadingOptions

@@assert_cardinality
@@at
@@bucket_by_sequence_length
@@cardinality
@@choose_from_datasets
@@copy_to_device
@@dense_to_ragged_batch
@@dense_to_sparse_batch
@@distribute
@@distributed_save
@@enable_debug_mode
@@enumerate_dataset
@@from_list
@@from_variant
@@get_model_proto
@@get_next_as_optional
@@get_single_element
@@get_structure
@@group_by_reducer
@@group_by_window
@@ignore_errors
@@index_table_from_dataset
@@load
@@make_batched_features_dataset
@@make_csv_dataset
@@make_saveable_from_iterator
@@map_and_batch
@@map_and_batch_with_legacy_function
@@pad_to_cardinality
@@parallel_interleave
@@parse_example_dataset
@@prefetch_to_device
@@rejection_resample
@@sample_from_datasets
@@save
@@scan
@@shuffle_and_repeat
@@snapshot
@@table_from_dataset
@@take_while
@@to_variant
@@unbatch
@@unique

@@AUTOTUNE
@@INFINITE_CARDINALITY
@@SHARD_HINT
@@UNKNOWN_CARDINALITY
"""

# pylint: disable=unused-import
from machina.python.data.experimental import service
from machina.python.data.experimental.ops.batching import dense_to_ragged_batch
from machina.python.data.experimental.ops.batching import dense_to_sparse_batch
from machina.python.data.experimental.ops.batching import map_and_batch
from machina.python.data.experimental.ops.batching import map_and_batch_with_legacy_function
from machina.python.data.experimental.ops.batching import unbatch
from machina.python.data.experimental.ops.cardinality import assert_cardinality
from machina.python.data.experimental.ops.cardinality import cardinality
from machina.python.data.experimental.ops.cardinality import INFINITE as INFINITE_CARDINALITY
from machina.python.data.experimental.ops.cardinality import UNKNOWN as UNKNOWN_CARDINALITY
from machina.python.data.experimental.ops.counter import Counter
from machina.python.data.experimental.ops.distribute import SHARD_HINT
from machina.python.data.experimental.ops.distributed_save_op import distributed_save
from machina.python.data.experimental.ops.enumerate_ops import enumerate_dataset
from machina.python.data.experimental.ops.error_ops import ignore_errors
from machina.python.data.experimental.ops.from_list import from_list
from machina.python.data.experimental.ops.get_single_element import get_single_element
from machina.python.data.experimental.ops.grouping import bucket_by_sequence_length
from machina.python.data.experimental.ops.grouping import group_by_reducer
from machina.python.data.experimental.ops.grouping import group_by_window
from machina.python.data.experimental.ops.grouping import Reducer
from machina.python.data.experimental.ops.interleave_ops import choose_from_datasets
from machina.python.data.experimental.ops.interleave_ops import parallel_interleave
from machina.python.data.experimental.ops.interleave_ops import sample_from_datasets
from machina.python.data.experimental.ops.io import load
from machina.python.data.experimental.ops.io import save
from machina.python.data.experimental.ops.iterator_model_ops import get_model_proto
from machina.python.data.experimental.ops.iterator_ops import make_saveable_from_iterator
from machina.python.data.experimental.ops.lookup_ops import DatasetInitializer
from machina.python.data.experimental.ops.lookup_ops import index_table_from_dataset
from machina.python.data.experimental.ops.lookup_ops import table_from_dataset
from machina.python.data.experimental.ops.pad_to_cardinality import pad_to_cardinality
from machina.python.data.experimental.ops.parsing_ops import parse_example_dataset
from machina.python.data.experimental.ops.prefetching_ops import copy_to_device
from machina.python.data.experimental.ops.prefetching_ops import prefetch_to_device
from machina.python.data.experimental.ops.random_access import at
from machina.python.data.experimental.ops.random_ops import RandomDataset
from machina.python.data.experimental.ops.readers import CsvDataset
from machina.python.data.experimental.ops.readers import make_batched_features_dataset
from machina.python.data.experimental.ops.readers import make_csv_dataset
from machina.python.data.experimental.ops.readers import SqlDataset
from machina.python.data.experimental.ops.resampling import rejection_resample
from machina.python.data.experimental.ops.scan_ops import scan
from machina.python.data.experimental.ops.shuffle_ops import shuffle_and_repeat
from machina.python.data.experimental.ops.snapshot import snapshot
from machina.python.data.experimental.ops.take_while_ops import take_while
from machina.python.data.experimental.ops.unique import unique
from machina.python.data.experimental.ops.writers import TFRecordWriter
from machina.python.data.ops.dataset_ops import AUTOTUNE
from machina.python.data.ops.dataset_ops import DatasetSpec as DatasetStructure
from machina.python.data.ops.dataset_ops import from_variant
from machina.python.data.ops.dataset_ops import get_structure
from machina.python.data.ops.dataset_ops import to_variant
from machina.python.data.ops.debug_mode import enable_debug_mode
from machina.python.data.ops.iterator_ops import get_next_as_optional
from machina.python.data.ops.optional_ops import Optional
from machina.python.data.ops.optional_ops import OptionalSpec as OptionalStructure
from machina.python.data.ops.options import AutoShardPolicy
from machina.python.data.ops.options import AutotuneAlgorithm
from machina.python.data.ops.options import AutotuneOptions
from machina.python.data.ops.options import DistributeOptions
from machina.python.data.ops.options import ExternalStatePolicy
from machina.python.data.ops.options import OptimizationOptions
from machina.python.data.ops.options import ThreadingOptions
from machina.python.data.util.structure import _RaggedTensorStructure as RaggedTensorStructure
from machina.python.data.util.structure import _SparseTensorStructure as SparseTensorStructure
from machina.python.data.util.structure import _TensorArrayStructure as TensorArrayStructure
from machina.python.data.util.structure import _TensorStructure as TensorStructure
from machina.python.framework.type_spec import TypeSpec as Structure
# pylint: enable=unused-import

from machina.python.util.all_util import remove_undocumented

_allowed_symbols = [
    "service",
]

remove_undocumented(__name__, _allowed_symbols)
