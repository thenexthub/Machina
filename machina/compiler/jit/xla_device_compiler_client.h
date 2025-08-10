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

#ifndef MACHINA_COMPILER_JIT_MACHINA_MACHINA_XLA_DEVICE_COMPILER_CLIENT_H_
#define MACHINA_COMPILER_JIT_MACHINA_MACHINA_XLA_DEVICE_COMPILER_CLIENT_H_

#include <memory>
#include <optional>
#include <string>

#include "machina/compiler/jit/device_compiler_client.h"
#include "machina/xla/client/local_client.h"

namespace machina {

class XlaDeviceCompilerClient
    : public DeviceCompilerClient<xla::LocalExecutable, xla::LocalClient> {
 public:
  explicit XlaDeviceCompilerClient(xla::LocalClient* client)
      : client_(client) {}

  absl::StatusOr<std::unique_ptr<xla::LocalExecutable>> BuildExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result) override;

  // Returns a serialized AOT result obtained by exporting the available
  // `executable` using the XlaCompiler.
  absl::StatusOr<std::string> SerializeExecutable(
      const xla::LocalExecutable& executable) override;

  // Returns a serialized AOT result obtained by compiling `result` into an AOT
  // result.
  absl::StatusOr<std::string> BuildSerializedExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result) override;

  // Loads a serialized AOT result (`serialized_executable`) into an
  // xla::LocalExecutable and returns it.
  absl::StatusOr<std::unique_ptr<xla::LocalExecutable>> LoadExecutable(
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& result,
      const std::string& serialized_executable) override;

  void WaitForProgramsToFinish() override;

  xla::LocalClient* client() const override { return client_; }

 private:
  xla::LocalClient* const client_;

  XlaDeviceCompilerClient(const XlaDeviceCompilerClient&) = delete;
  void operator=(const XlaDeviceCompilerClient&) = delete;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_JIT_MACHINA_MACHINA_XLA_DEVICE_COMPILER_CLIENT_H_
