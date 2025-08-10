/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include "machina/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.h"

#include <cassert>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "machina/xla/stream_executor/stream.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/refcount.h"
#include "machina/core/platform/status.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

static void ReportInternalError(machina::OpKernelContext *ctx,
                                const std::string msg) {
  if (ctx == nullptr) {
    LOG(WARNING) << msg << "\n";
    return;
  }
  ctx->CtxFailureWithWarning(absl::Status{absl::StatusCode::kInternal, msg});
}

#if GOOGLE_CUDA
using GPUResult = CUresult;
#endif
#if MACHINA_USE_ROCM
using GPUResult = hipError_t;
#endif

void GPUReportIfError(GPUResult result, machina::OpKernelContext *ctx,
                      const char *expr_str) {
  if (!result) return;
  const char *name = nullptr;

#if GOOGLE_CUDA
  cuGetErrorName(result, &name);
#endif
#if MACHINA_USE_ROCM
  name = hipGetErrorName(result);
#endif

  if (!name) name = "<unknown>";
  std::string msg = absl::StrCat("'", expr_str, "' failed with '", name, "'");
  ReportInternalError(ctx, msg);
}

#define GPU_REPORT_IF_ERROR_WITH_CTX(expr, ctx) \
  GPUReportIfError(expr, ctx, #expr)
#define GPU_REPORT_IF_ERROR(expr) GPU_REPORT_IF_ERROR_WITH_CTX(expr, nullptr)

// Implement the GPU module cache and share what can be shared.

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

GPURuntimeCache::~GPURuntimeCache() {
  machina::mutex_lock lock(mu_);
  for (auto it : gpu_module_by_data_ptr_) {
#if GOOGLE_CUDA
    GPU_REPORT_IF_ERROR(cuModuleUnload(it.second));
#endif
#if MACHINA_USE_ROCM
    GPU_REPORT_IF_ERROR(hipModuleUnload(it.second));
#endif
  }
}

absl::Status GPURuntimeCache::Create(GPURuntimeCache **dst) {
  *dst = new GPURuntimeCache;
  return absl::OkStatus();
}

std::string GPURuntimeCache::DebugString() const { return "GPU runtime cache"; }

GPURuntimeCache::GPUModule GPURuntimeCache::LookupOrLoadModule(void *data) {
  machina::mutex_lock lock(mu_);
  GPUModule &module = gpu_module_by_data_ptr_[data];

#if GOOGLE_CUDA
  if (!module) GPU_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
#endif
#if MACHINA_USE_ROCM
  if (!module) GPU_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
#endif

  return module;
}

GPURuntimeCache::GPUFunction GPURuntimeCache::LookupOrGetFunction(
    GPUModule module, const char *kernel_name) {
  machina::mutex_lock lock(mu_);
  GPUFunction &function =
      gpu_function_by_module_and_name_[{module, kernel_name}];

  if (!function) {
#if GOOGLE_CUDA
    GPU_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, kernel_name));
#endif
#if MACHINA_USE_ROCM
    GPU_REPORT_IF_ERROR(hipModuleGetFunction(&function, module, kernel_name));
#endif
  }

  return function;
}

// Implements a C wrapper around the TensorFlow runtime and CUDA (or ROCm)
// library that allows launching a kernel on the current device and stream from
// a binary blob for the module and function name.
// The wrapper uses intptr_t instead of CUDA's unsigned int (or ROCm's unsigned
// int) to match the type of MLIR's index type. This avoids the need for casts
// in the generated MLIR code.
extern "C" void _mlir_ciface_tf_launch_kernel(void *ctx, void *module_blob,
                                              char *kernel_name, intptr_t gridX,
                                              intptr_t gridY, intptr_t gridZ,
                                              intptr_t blockX, intptr_t blockY,
                                              intptr_t blockZ, void **params) {
  // For empty grids, we don't need to do anything.
  if (!gridX || !gridY || !gridZ) return;

  // Get the GPU module cache.
  auto *op_kernel_ctx = static_cast<machina::OpKernelContext *>(ctx);
  auto *rm = op_kernel_ctx->resource_manager();
  if (rm == nullptr) {
    ReportInternalError(op_kernel_ctx, "expected resource_manager");
    return;
  }
  GPURuntimeCache *cache = nullptr;
  OP_REQUIRES_OK(op_kernel_ctx, rm->LookupOrCreate<GPURuntimeCache>(
                                    rm->default_container(),
                                    GPURuntimeCache::kDefaultResourceName,
                                    &cache, GPURuntimeCache::Create));
  assert(cache != nullptr && "cache creation must not fail");
  machina::core::ScopedUnref ref(cache);

  // Get the GPU module.
  stream_executor::Stream *se_stream =
      op_kernel_ctx->op_device_context()->stream();
  void *stream = se_stream->platform_specific_handle().stream;
  GPURuntimeCache::GPUModule module = cache->LookupOrLoadModule(module_blob);
  GPURuntimeCache::GPUFunction function =
      cache->LookupOrGetFunction(module, kernel_name);

#if GOOGLE_CUDA
  GPU_REPORT_IF_ERROR_WITH_CTX(
      cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                     /*sharedMemBytes=*/0, reinterpret_cast<CUstream>(stream),
                     params, nullptr),
      op_kernel_ctx);
#endif
#if MACHINA_USE_ROCM
  GPU_REPORT_IF_ERROR_WITH_CTX(
      hipModuleLaunchKernel(
          function, gridX, gridY, gridZ, blockX, blockY, blockZ,
          /*sharedMemBytes=*/0, reinterpret_cast<hipStream_t>(stream), params,
          nullptr),
      op_kernel_ctx);
#endif
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
