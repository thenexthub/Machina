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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_DATA_DUMPER_LOGGER_CONFIG_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_DATA_DUMPER_LOGGER_CONFIG_H_

#include <functional>
#include <string>

#include "machina/compiler/mlir/machina/utils/bridge_logger.h"

namespace machina {

class DataDumperLoggerConfig : public ::machina::BridgeLoggerConfig {
 public:
  explicit DataDumperLoggerConfig(
      std::function<std::string(const std::string &, mlir::Operation *op)>
          get_filename,
      const std::string &pass_prefix = "", bool print_module_scope = false,
      bool print_after_only_on_change = true,
      mlir::OpPrintingFlags op_printing_flags = mlir::OpPrintingFlags());

  void printBeforeIfEnabled(mlir::Pass *pass, mlir::Operation *op,
                            PrintCallbackFn print_callback) override;

  void printAfterIfEnabled(mlir::Pass *pass, mlir::Operation *op,
                           PrintCallbackFn print_callback) override;

 private:
  static void DumpMlir(const std::string &filename,
                       BridgeLoggerConfig::PrintCallbackFn print_callback);

  // The function to dump the target MLIR string to file.
  // The parameter that will be sent to the dump_func_ is:
  // The pass name (std::string)
  std::function<std::string(const std::string &, mlir::Operation *op)>
      get_filename_;

  // The pass prefix.
  std::string pass_prefix_;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_DATA_DUMPER_LOGGER_CONFIG_H_
