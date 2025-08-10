/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include <memory>
#include <utility>
#include <vector>

#include "machina/compiler/jit/device_compilation_profiler.h"
#include "machina/compiler/jit/flags.h"
#include "machina/compiler/jit/device_compiler.h"
#include "machina/compiler/jit/xla_device_compiler_client.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/xla/client/client_library.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

// This test is kept separate because it disables XLA compilation globally.
TEST(DeviceCompilerTest, TestDisabledXlaCompilation) {
  NameAttrList fn;
  fn.set_name("afunction");

  // Create mock arguments so we see them in the VLOG when compilation fails.
  std::vector<XlaCompiler::Argument> args(2);
  for (int i = 0; i < 2; ++i) {
    args[i].kind = XlaCompiler::Argument::kParameter;
    args[i].type = DT_INT32;
    args[i].shape = TensorShape({2, i + 1});
    args[i].name = absl::StrCat("arg", i);
  }

  DisableXlaCompilation();

  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  DeviceType device_type = DeviceType(DEVICE_CPU_MACHINA_MACHINA_XLA_JIT);

  const XlaCompiler::CompilationResult* compilation_result;
  xla::LocalExecutable* executable;

  using XlaDeviceExecutablePersistor =
      DeviceExecutablePersistor<xla::LocalExecutable, xla::LocalClient>;
  auto persistor = std::make_unique<XlaDeviceExecutablePersistor>(
      XlaDeviceExecutablePersistor::Config(), device_type);
  auto compiler_client = std::make_unique<XlaDeviceCompilerClient>(client);
  auto xla_device_compiler =
      new DeviceCompiler<xla::LocalExecutable, xla::LocalClient>(
          std::move(persistor), std::move(compiler_client));
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  auto profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  // Check that strict compilation is disallowed.
  absl::Status status = xla_device_compiler->CompileIfNeeded(
      XlaCompiler::Options{}, fn, args, XlaCompiler::CompileOptions{},
      DeviceCompileMode::kStrict, profiler, &compilation_result, &executable);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.message(), "XLA compilation disabled"));

  // Check that async compilation is disallowed.
  status = xla_device_compiler->CompileIfNeeded(
      XlaCompiler::Options{}, fn, args, XlaCompiler::CompileOptions{},
      DeviceCompileMode::kAsync, profiler, &compilation_result, &executable);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.message(), "XLA compilation disabled"));

  // Check that lazy compilation is disallowed.
  status = xla_device_compiler->CompileIfNeeded(
      XlaCompiler::Options{}, fn, args, XlaCompiler::CompileOptions{},
      DeviceCompileMode::kLazy, profiler, &compilation_result, &executable);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.message(), "XLA compilation disabled"));
}

}  // namespace
}  // namespace machina
