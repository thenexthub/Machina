###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Friday, April 11, 2025.                                             #
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
"""Stub to make pywrap metrics wrapper accessible."""

from machina.compiler.mlir.lite.metrics import converter_error_data_pb2
from machina.compiler.mlir.lite.python import wrap_converter
from machina.lite.python.metrics._pywrap_machina_lite_metrics_wrapper import MetricsWrapper  # pylint: disable=unused-import


def retrieve_collected_errors():
  """Returns and clears the list of collected errors in ErrorCollector.

  The RetrieveCollectedErrors function in C++ returns a list of serialized proto
  messages. This function will convert them to ConverterErrorData instances.

  Returns:
    A list of ConverterErrorData.
  """
  serialized_message_list = wrap_converter.wrapped_retrieve_collected_errors()
  return list(
      map(converter_error_data_pb2.ConverterErrorData.FromString,
          serialized_message_list))
