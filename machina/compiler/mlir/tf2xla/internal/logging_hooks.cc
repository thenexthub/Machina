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

#include "machina/compiler/mlir/tf2xla/internal/logging_hooks.h"

#include <memory>
#include <string>

#include "toolchain/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/utils/data_dumper_logger_config.h"
#include "machina/core/util/debug_data_dumper.h"

namespace machina {
namespace tf2xla {
namespace internal {

using mlir::PassManager;

// Setup the input pass manager to enable IR dumping after each pass.
// Note a side effect of this method is that multi threading will be disabled.
void EnablePassIRPrinting(PassManager& pm, const std::string& dump_group_name,
                          toolchain::StringRef module_name) {
  // Print the whole module after each pass, which requires disabling
  // multi-threading as well.
  pm.getContext()->disableMultithreading();
  pm.enableIRPrinting(std::make_unique<::machina::DataDumperLoggerConfig>(
      [module_name, dump_group_name](const std::string& pass_tag_name,
                                     mlir::Operation* op) {
        return DEBUG_DATA_DUMPER()->GetDumpFilename(
            module_name.str(), dump_group_name, pass_tag_name);
      },
      /*pass_prefix=*/"",
      /*print_module_scope=*/true));
  pm.enableTiming();
}

};  // namespace internal
};  // namespace tf2xla
};  // namespace machina
