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
"""Simple benchmarks for reductions and their gradients."""

import time

import numpy as np

from machina.core.protobuf import config_pb2
from machina.python.client import session
from machina.python.eager import backprop
from machina.python.eager import context
from machina.python.framework import constant_op
from machina.python.framework import ops
from machina.python.ops import array_ops
from machina.python.ops import gradients_impl
from machina.python.ops import math_ops
from machina.python.platform import test


class ReduceBenchmarks(test.Benchmark):
  """Benchmarks for reductions."""

  def _run(self, func, num_iters):
    # call func to maybe warm up the GPU
    func()
    start = time.time()
    for _ in range(num_iters):
      func()
    end = time.time()
    mean_us = (end - start) * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={"examples_per_sec": num_iters / (end - start)})

  def benchmark_reduce_sum_grad_eager(self):
    with context.eager_mode():
      tensor = array_ops.zeros([100, 1000])

      def fn():
        backprop.gradients_function(math_ops.reduce_sum, [0])(tensor)

      self._run(fn, 10000)

  def benchmark_reduce_sum_grad_eager_cpu(self):
    with context.eager_mode(), ops.device("/cpu:0"):
      tensor = array_ops.zeros([100, 1000])

      def fn():
        backprop.gradients_function(math_ops.reduce_sum, [0])(tensor)

      self._run(fn, 10000)

  def benchmark_reduce_sum_grad_graph(self):
    config = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0)))
    with ops.Graph().as_default(), session.Session(config=config) as sess:

      tensor = constant_op.constant(np.zeros([100, 1000], dtype=np.float32))
      reduction = math_ops.reduce_sum(tensor)
      grad, = gradients_impl.gradients(reduction, tensor)

      def fn():
        self.evaluate(grad.op)

      self._run(fn, 10000)

  def benchmark_reduce_sum_grad_graph_cpu(self):
    config = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0)))
    with ops.Graph().as_default(), session.Session(config=config) as sess:

      with ops.device("/cpu:0"):
        tensor = constant_op.constant(np.zeros([100, 1000], dtype=np.float32))
        reduction = math_ops.reduce_sum(tensor)
        grad, = gradients_impl.gradients(reduction, tensor)

      def fn():
        self.evaluate(grad.op)

      self._run(fn, 10000)


if __name__ == "__main__":
  test.main()
