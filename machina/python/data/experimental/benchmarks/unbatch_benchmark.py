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
"""Benchmarks for `tf.data.Dataset.unbatch()`."""
from machina.python.data.benchmarks import benchmark_base
from machina.python.data.ops import dataset_ops


class UnbatchBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.Dataset.unbatch()`."""

  def benchmark_native_unbatch(self):
    batch_sizes = [1, 2, 5, 10, 20, 50]
    num_elements = 10000

    for batch_size in batch_sizes:
      dataset = dataset_ops.Dataset.from_tensors("element").repeat(None)
      dataset = dataset.batch(batch_size)
      dataset = dataset.unbatch()

      self.run_and_report_benchmark(
          dataset=dataset,
          num_elements=num_elements,
          iters=5,
          extras={
              "model_name": "unbatch.benchmark.1",
              "parameters": "%d" % batch_size,
          },
          name="native_batch_size_%d" % batch_size)

  # Include a benchmark of the previous `unbatch()` implementation that uses
  # a composition of more primitive ops. Eventually we'd hope to generate code
  # that is as good in both cases.
  def benchmark_old_unbatch_implementation(self):
    batch_sizes = [1, 2, 5, 10, 20, 50]
    num_elements = 10000

    for batch_size in batch_sizes:
      dataset = dataset_ops.Dataset.from_tensors("element").repeat(None)
      dataset = dataset.batch(batch_size)
      dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)

      self.run_and_report_benchmark(
          dataset=dataset,
          num_elements=num_elements,
          iters=5,
          extras={
              "model_name": "unbatch.benchmark.2",
              "parameters": "%d" % batch_size,
          },
          name="unfused_batch_size_%d" % batch_size)


if __name__ == "__main__":
  benchmark_base.test.main()
