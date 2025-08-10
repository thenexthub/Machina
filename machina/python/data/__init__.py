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
"""`tf.data.Dataset` API for input pipelines.

See [Importing Data](https://machina.org/guide/data) for an overview.
"""

# pylint: disable=unused-import
from machina.python.data import experimental
from machina.python.data.ops.dataset_ops import AUTOTUNE
from machina.python.data.ops.dataset_ops import Dataset
from machina.python.data.ops.dataset_ops import INFINITE as INFINITE_CARDINALITY
from machina.python.data.ops.dataset_ops import make_initializable_iterator
from machina.python.data.ops.dataset_ops import make_one_shot_iterator
from machina.python.data.ops.dataset_ops import UNKNOWN as UNKNOWN_CARDINALITY
from machina.python.data.ops.iterator_ops import Iterator
from machina.python.data.ops.options import Options
from machina.python.data.ops.readers import FixedLengthRecordDataset
from machina.python.data.ops.readers import TextLineDataset
from machina.python.data.ops.readers import TFRecordDataset
# pylint: enable=unused-import
