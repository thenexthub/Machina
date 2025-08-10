/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/compiler/mlir/machina/utils/data_dumper_logger_config.h"

#include <functional>
#include <memory>
#include <string>

#include "machina/compiler/mlir/machina/utils/dump_mlir_util.h"

namespace machina {
DataDumperLoggerConfig::DataDumperLoggerConfig(
    std::function<std::string(const std::string &, mlir::Operation *op)>
        get_filename,
    const std::string &pass_prefix, bool print_module_scope,
    bool print_after_only_on_change, mlir::OpPrintingFlags op_printing_flags)
    : ::machina::BridgeLoggerConfig(
          print_module_scope, print_after_only_on_change, op_printing_flags),
      get_filename_(get_filename),
      pass_prefix_(pass_prefix) {}

void DataDumperLoggerConfig::printBeforeIfEnabled(
    mlir::Pass *pass, mlir::Operation *op, PrintCallbackFn print_callback) {
  std::string pass_name = pass->getName().str();
  std::string filename =
      get_filename_(pass_prefix_ + "before_" + pass_name, op);

  if (ShouldPrint(pass, op)) DumpMlir(filename, print_callback);
}

void DataDumperLoggerConfig::printAfterIfEnabled(
    mlir::Pass *pass, mlir::Operation *op, PrintCallbackFn print_callback) {
  std::string pass_name = pass->getName().str();
  std::string filename = get_filename_(pass_prefix_ + "after_" + pass_name, op);

  if (ShouldPrint(pass, op)) DumpMlir(filename, print_callback);
}

void DataDumperLoggerConfig::DumpMlir(
    const std::string &filename,
    BridgeLoggerConfig::PrintCallbackFn print_callback) {
  std::unique_ptr<toolchain::raw_ostream> os;
  std::string filepath;
  if (machina::CreateFileForDumping(filename, &os, &filepath).ok()) {
    print_callback(*os);
    LOG(INFO) << "Dumped MLIR module to " << filepath;
  }
}
}  // namespace machina
