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
"""Test for the tf.test.benchmark."""

import os
from google.protobuf import json_format
from machina.core.util import test_log_pb2
from machina.python.platform import benchmark
from machina.python.platform import test

class BenchmarkTest(test.TestCase, benchmark.TensorFlowBenchmark):

  def testReportBenchmark(self):
    output_dir = self.get_temp_dir() + os.path.sep
    os.environ['TEST_REPORT_FILE_PREFIX'] = output_dir
    proto_file_path = os.path.join(output_dir,
                                   'BenchmarkTest.testReportBenchmark')
    if os.path.exists(proto_file_path):
      os.remove(proto_file_path)

    self.report_benchmark(
        iters=2000,
        wall_time=1000,
        name='testReportBenchmark',
        metrics=[{'name': 'metric_name_1', 'value': 0, 'min_value': 1},
                 {'name': 'metric_name_2', 'value': 90, 'min_value': 0,
                  'max_value': 95}])

    with open(proto_file_path, 'rb') as f:
      benchmark_entries = test_log_pb2.BenchmarkEntries()
      benchmark_entries.ParseFromString(f.read())

      actual_result = json_format.MessageToDict(
          benchmark_entries, preserving_proto_field_name=True,
          always_print_fields_with_no_presence=True)['entry'][0]
    os.remove(proto_file_path)

    expected_result = {
        'name': 'BenchmarkTest.testReportBenchmark',
        # google.protobuf.json_format.MessageToDict() will convert
        # int64 field to string.
        'iters': '2000',
        'wall_time': 1000,
        'cpu_time': 0,
        'throughput': 0,
        'extras': {},
        'metrics': [
            {
                'name': 'metric_name_1',
                'value': 0,
                'min_value': 1
            },
            {
                'name': 'metric_name_2',
                'value': 90,
                'min_value': 0,
                'max_value': 95
            }
        ]
    }

    self.assertEqual(2000, benchmark_entries.entry[0].iters)
    self.assertDictEqual(expected_result, actual_result)

if __name__ == '__main__':
  test.main()
