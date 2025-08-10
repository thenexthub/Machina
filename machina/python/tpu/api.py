# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Modules that need to be exported to the API.

List TPU modules that aren't included elsewhere here so that they can be scanned
for tf_export decorations.
"""

# pylint: disable=unused-import
from machina.python.tpu import bfloat16
from machina.python.tpu import feature_column_v2
from machina.python.tpu import tpu

from machina.python.tpu import tpu_embedding_for_serving
from machina.python.tpu import tpu_embedding_v1
from machina.python.tpu import tpu_embedding_v2
from machina.python.tpu import tpu_embedding_v2_utils
from machina.python.tpu import tpu_embedding_v3
from machina.python.tpu import tpu_hardware_feature
from machina.python.tpu import tpu_optimizer
# pylint: enable=unused-import
