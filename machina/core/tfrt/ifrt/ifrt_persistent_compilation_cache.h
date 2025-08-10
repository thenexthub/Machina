/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#ifndef MACHINA_CORE_TFRT_IFRT_IFRT_PERSISTENT_COMPILATION_CACHE_H_
#define MACHINA_CORE_TFRT_IFRT_IFRT_PERSISTENT_COMPILATION_CACHE_H_

#include <memory>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "machina/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "machina/xla/pjrt/pjrt_executable.h"
#include "machina/xla/python/ifrt/device_list.h"
#include "machina/xla/python/ifrt/executable.h"
#include "machina/xla/python/ifrt/hlo/hlo_program.h"
#include "machina/xla/python/ifrt/host_callback.h"
#include "machina/xla/python/ifrt/program.h"
#include "machina/xla/tsl/concurrency/ref_count.h"
namespace machina {
namespace ifrt_serving {

class IfrtPersistentCompilationCache {
 public:
  IfrtPersistentCompilationCache() = default;
  virtual ~IfrtPersistentCompilationCache() = default;

  // The implementation of this API should be thread-safe. It generates a key
  // for looking up the executable in the persistent cache and it will return
  // the LoadedExecutable if hits cache. Otherwise, it will call the `value_fn`
  // to generate and return the LoadedExecutable.
  virtual absl::StatusOr<xla::ifrt::LoadedExecutableRef>
  LookupLoadedExecutableOrCreate(
      std::unique_ptr<xla::ifrt::HloProgram> hlo_program,
      xla::ifrt::DeviceListRef device_list,
      const xla::CompileOptions& xla_compile_options,
      const std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>&
          loaded_host_callbacks,
      xla::ifrt::Client* client,
      absl::AnyInvocable<absl::StatusOr<xla::ifrt::LoadedExecutableRef>(
          std::unique_ptr<xla::ifrt::Program> program,
          std::unique_ptr<xla::ifrt::CompileOptions> options)>
          value_fn);

  // The implementation of this API should be thread-safe. It generates a key
  // for looking up the Tf2HloResult in the persistent cache and it will return
  // the Tf2HloResult if hits cache. Otherwise, it will call the `value_fn` to
  // generate and return the Tf2HloResult.
  virtual absl::StatusOr<Tf2HloResult> LookupTf2HloResultOrCreate(
      Tf2HloArg tf2hlo_arg, TfToHloCompiler* tf_to_hlo_compiler);

  virtual bool IsXlaCompilationCacheEnabled() const { return false; }
  virtual bool IsTf2HloCompilationCacheEnabled() const { return false; }
};

}  // namespace ifrt_serving
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_IFRT_IFRT_PERSISTENT_COMPILATION_CACHE_H_
