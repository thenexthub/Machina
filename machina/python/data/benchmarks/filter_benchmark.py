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
"""Benchmarks for `tf.data.Dataset.filter()`."""
from machina.python.data.benchmarks import benchmark_base
from machina.python.data.ops import dataset_ops
from machina.python.ops import array_ops


class FilterBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.Dataset.filter()`."""

  def _benchmark(self, predicate, name, benchmark_id):
    num_elements = 100000
    dataset = dataset_ops.Dataset.from_tensors(True)
    dataset = dataset.repeat().filter(predicate)
    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        extras={
            "model_name": "filter.benchmark.%d" % benchmark_id,
            "parameters": "%d" % num_elements,
        },
        name=name)

  def benchmark_simple_function(self):
    self._benchmark(array_ops.identity, "simple_function", benchmark_id=1)

  def benchmark_return_component_optimization(self):
    self._benchmark(lambda x: x, "return_component", benchmark_id=2)


if __name__ == "__main__":
  benchmark_base.test.main()
