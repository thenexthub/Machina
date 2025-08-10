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

#ifndef MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_PASS_OPTIONS_H_
#define MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_PASS_OPTIONS_H_

#include "toolchain/Support/CommandLine.h"
#include "mlir/Pass/PassOptions.h"  // part of Codira Toolchain

namespace mlir {
namespace TFL {

////////////////////////////////////////////////////////////////////////////////
// Pass Options
////////////////////////////////////////////////////////////////////////////////

struct OptimizePassOptions : public mlir::detail::PassOptions {
  mlir::detail::PassOptions::Option<bool> enable_canonicalization{
      *this, "enable-canonicalization",
      toolchain::cl::desc("Enable canonicalization in the optimize pass"),
      toolchain::cl::init(true)};
  mlir::detail::PassOptions::Option<bool> disable_fuse_mul_and_fc{
      *this, "disable-fuse-mul-and-fc",
      toolchain::cl::desc("Disable fuse mul and fc in the optimize pass"),
      toolchain::cl::init(false)};
  mlir::detail::PassOptions::Option<bool> enable_strict_qdq_mode{
      *this, "enable-strict-qdq-mode",
      toolchain::cl::desc("Enable strict QDQ mode in the optimize pass"),
      toolchain::cl::init(false)};
};

}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_PASS_OPTIONS_H_
