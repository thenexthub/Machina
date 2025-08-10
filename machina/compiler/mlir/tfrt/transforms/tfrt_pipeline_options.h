/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_TFRT_PIPELINE_OPTIONS_H_
#define MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_TFRT_PIPELINE_OPTIONS_H_

#include <cstdint>
#include <string>

#include "toolchain/Support/CommandLine.h"
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfrt/translate/tfrt_compile_options.h"

namespace machina {

struct TfrtPipelineOptions
    : public mlir::PassPipelineOptions<TfrtPipelineOptions> {
  Option<std::string> saved_model_dir{*this, "saved-model-dir",
                                      toolchain::cl::desc(""), toolchain::cl::init("")};
  Option<std::string> default_device{
      *this, "default-device", toolchain::cl::desc("default device assignment"),
      toolchain::cl::init("/job:localhost/replica:0/task:0/device:CPU:0")};
  Option<bool> enable_optimizer{
      *this, "enable-optimizer",
      toolchain::cl::desc("run optimization passes on corert dialect"),
      toolchain::cl::init(false)};
  Option<bool> decompose_resource_ops{
      *this, "decompose-resource-ops",
      toolchain::cl::desc("decompose composite resource ops into ReadVariableOp and "
                     "non-resource ops. This is currently used in TFRT "
                     "savedmodel pipeline."),
      toolchain::cl::init(false)};
  Option<std::string> force_data_format{
      *this, "force-data-format",
      toolchain::cl::desc("force data format for all layout sensitive operations")};
  // TODO(tfrt-devs): consider making compiler to figure out whether to fold
  // transpose or not instead of exposing the specific option.
  Option<bool> skip_fold_transpose_in_ops{
      *this, "skip-fold-transpose-in-ops",
      toolchain::cl::desc("Skip folding transpose operands in Ops which can support "
                     "different layouts.")};
  Option<bool> target_tpurt{*this, "target-tpurt",
                            toolchain::cl::desc("target TPURT dialect if true"),
                            toolchain::cl::init(false)};
  Option<bool> tpu_use_core_selector{
      *this, "tpu-use-core-selector",
      toolchain::cl::desc("If true, use ServingCoreSelector to pick TPU core. "
                     "Otherwise, use the assigned core. Currently we use "
                     "core selector for Servo serving use cases."),
      toolchain::cl::init(true)};
  Option<bool> tpu_use_bundled_transfer{
      *this, "tpu-use-bundled-transfer",
      toolchain::cl::desc("If true, use BundledTransferToTpuOp to transfer "
                     "variables and input tensors to TPU."),
      toolchain::cl::init(true)};
  Option<bool> tpu_lower_to_fallback{
      *this, "tpu-lower-to-fallback",
      toolchain::cl::desc("If true, lower an TF op that's placed on TPU device "
                     "to be executed by tfrt_fallback.execute."),
      toolchain::cl::init(true)};
  Option<bool> tpu_fuse_ops{
      *this, "tpu-fuse-ops",
      toolchain::cl::desc("If true, use the TPU fused compile_and_execute kernel"),
      toolchain::cl::init(false)};
  // TODO(b/194081364): remove this option once we unify servo TPU serving
  // result transfer behavior.
  Option<bool> tpu_transfer_result_to_host{
      *this, "tpu-transfer-result-to-host",
      toolchain::cl::desc("If true, transfer the result of tpurt.execute from TPU "
                     "to host."),
      toolchain::cl::init(true)};
  Option<bool> use_tpu_host_allocator_for_inputs{
      *this, "use-tpu-host-allocator-for-inputs",
      toolchain::cl::desc("If true, fallback executeops that produce inputs to tpu "
                     "program will use tpu host allocator."),
      toolchain::cl::init(false)};
  Option<TfrtCompileOptions::TpuAllowUnpaddedBatch> tpu_allow_unpadded_batch{
      *this, "tpu-allow-unpadded-batch",
      toolchain::cl::desc("To allow unpadded batch for TPU execution."),
      toolchain::cl::values(
          clEnumValN(TfrtCompileOptions::TpuAllowUnpaddedBatch::kDisabled,
                     "disabled", "Disable this feature."),
          clEnumValN(TfrtCompileOptions::TpuAllowUnpaddedBatch::kAuto, "auto",
                     "Enable this feature when in-graph batching is detected."),
          clEnumValN(TfrtCompileOptions::TpuAllowUnpaddedBatch::kEnforced,
                     "enforced", "Force to enable this feature.")),
      toolchain::cl::init(TfrtCompileOptions::TpuAllowUnpaddedBatch::kDisabled)};

  Option<bool> target_gpu{
      *this, "target-gpu",
      toolchain::cl::desc("If true, target GPU compiler passes."),
      toolchain::cl::init(false)};

  // TODO(b/294895431): Remove the flag and default to the fused op.
  Option<bool> use_gpu_compile_and_execute_op{
      *this, "use-gpu-compile-and-execute-op",
      toolchain::cl::desc("If true, gpurt.compile_and_execute is used for GPU"),
      toolchain::cl::init(false)};

  Option<bool> enable_while_parallel_iterations{
      *this, "enable-while-parallel-iterations",
      toolchain::cl::desc("If true, tf.While op will be parallelized. This is "
                     "currently experimental."),
      toolchain::cl::init(false)};

  Option<bool> hoist_invariant_ops{
      *this, "hoist-invariant-ops",
      toolchain::cl::desc("If true, invariant ops in savedmodels will be hoisted "
                     "out to run during loading."),
      toolchain::cl::init(false)};

  Option<bool> fuse_get_resource_ops_in_hoisting{
      *this, "fuse-get-resource-ops-in-hoisting",
      toolchain::cl::desc("If true, get_resource_op will be fused during hoisting"),
      toolchain::cl::init(true)};

  Option<bool> sink_in_invariant_ops{
      *this, "sink-in-invariant-ops",
      toolchain::cl::desc("If true, sink the selected invariant ops in to the "
                     "nested functions to facilitate invariant ops hoisting."),
      toolchain::cl::init(false)};

  Option<uint64_t> cost_threshold{
      *this, "tfrt-cost-threshold",
      toolchain::cl::desc(
          "The cost threshold to decide whether a sequence of operations is "
          "cheap, and then whether it can be executed inline."),
      toolchain::cl::init(1)};

  Option<int64_t> min_num_batch_threads{
      *this, "tfrt-min-num-batch-threads",
      toolchain::cl::desc("The minimum number of batch threads"), toolchain::cl::init(1)};

  Option<int64_t> min_max_enqueued_batches{
      *this, "tfrt-min-max-enqueued-batches",
      toolchain::cl::desc(
          "The minimum of the maximum number of outstanding enqueued batches"),
      toolchain::cl::init(1)};

  Option<int64_t> batch_queue_global_prioritization_num_threads{
      *this, "tfrt-batch-queue-global-prioritization-num-threads",
      toolchain::cl::desc(
          "If non-zero, all models on this server are switched to use a "
          "prioritized batching function using this number of global threads."),
      toolchain::cl::init(0)};

  Option<std::string> batch_padding_policy{
      *this, "tfrt-batch-padding-policy",
      toolchain::cl::desc("The policy used when padding (or splitting) batches."),
      toolchain::cl::init("")};

  Option<int64_t> num_batch_threads{
      *this, "tfrt-num-batch-threads",
      toolchain::cl::desc(
          "The number of threads for processing batches in parallel"),
      toolchain::cl::init(0)};

  Option<int64_t> max_batch_size{
      *this, "tfrt-max-batch-size",
      toolchain::cl::desc("The maximum allowed batch size"), toolchain::cl::init(0)};

  Option<int64_t> batch_timeout_micros{
      *this, "tfrt-batch-timeout-micros",
      toolchain::cl::desc("The maximum number of microseconds before outputting an "
                     "incomplete batch"),
      toolchain::cl::init(0)};

  ListOption<int64_t> allowed_batch_sizes{
      *this, "tfrt-allowed-batch-sizes",
      toolchain::cl::desc("Allowed sizes for padding (or splitting) batches")};

  Option<int64_t> max_enqueued_batches{
      *this, "tfrt-max-enqueued-batches",
      toolchain::cl::desc("The maximum number of batches enqueued for processing "
                     "before requests are failed fast"),
      toolchain::cl::init(0)};

  Option<bool> merge_inter_dependent_streams{
      *this, "tfrt-merge-inter-dependent-streams",
      toolchain::cl::desc("If true, streams with inter data depenedencies will be "
                     "preferred to be merged for inline execution."),
      toolchain::cl::init(false)};
};

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_TFRT_PIPELINE_OPTIONS_H_
