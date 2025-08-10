/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_COMPILER_JIT_PJRT_DEVICE_COMPILER_CLIENT_H_
#define MACHINA_COMPILER_JIT_PJRT_DEVICE_COMPILER_CLIENT_H_

#include <memory>
#include <optional>
#include <string>

#include "machina/compiler/jit/device_compiler_client.h"
#include "machina/xla/pjrt/pjrt_client.h"

namespace machina {

// Calls into PjRtClient to provide functionality for building, serializing and
// loading PjRtLoadedExecutables.
class PjRtDeviceCompilerClient
    : public DeviceCompilerClient<xla::PjRtLoadedExecutable, xla::PjRtClient> {
 public:
  explicit PjRtDeviceCompilerClient(xla::PjRtClient* client)
      : client_(client) {}

  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> BuildExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result) override;

  // Returns a platform-specific serialization of `executable`. The
  // serialization is not guaranteed to be stable over time. `executable` must
  // have been produced by this client.
  absl::StatusOr<std::string> SerializeExecutable(
      const xla::PjRtLoadedExecutable& executable) override;

  // PjRt doesn't support AOT compilation yet. Builds a PjRtLoadedExecutable and
  // serializes it to string.
  absl::StatusOr<std::string> BuildSerializedExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result) override;

  // Deserializes a serialized executable as produced by
  // PjRtExecutable::SerializeExecutable(). `serialized_executable` must have
  // been produced by a compiler of the same platform and version as this one.
  //
  // PjRt doesn't support AOT compilation yet. Loading a serialized executable
  // is currently only implemented for TfrtTpuPjrtClient and hence, this
  // function doesn't use PjRtClient::LoadSerializedExecutable() and uses
  // PjRtClient::DeserializeExecutable() instead.
  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> LoadExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result,
      const std::string& serialized_executable) override;

  // No-op. PJRT uses futures and waiting for programs to finish isn't
  // necessary.
  void WaitForProgramsToFinish() override;

  xla::PjRtClient* client() const override { return client_; }

 private:
  xla::PjRtClient* const client_;

  PjRtDeviceCompilerClient(const PjRtDeviceCompilerClient&) = delete;
  void operator=(const PjRtDeviceCompilerClient&) = delete;
};

// Generates CompileOptions for PJRT compilation.
xla::CompileOptions GetPjRtCompileOptions(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result);

}  // namespace machina

#endif  // MACHINA_COMPILER_JIT_PJRT_DEVICE_COMPILER_CLIENT_H_
