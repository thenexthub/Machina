/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "tfrt/bef_converter/bef_to_mlir_translate.h"  // from @tf_runtime
#include "tfrt/bef_converter/mlir_to_bef_translate.h"  // from @tf_runtime
#include "tfrt/init_tfrt_dialects.h"  // from @tf_runtime

static mlir::TranslateFromMLIRRegistration mlir_to_bef_registration(
    "mlir-to-bef", "translate MLIR to BEF", tfrt::MLIRToBEFTranslate,
    [](mlir::DialectRegistry &registry) {
      tfrt::RegisterTFRTDialects(registry);
      tfrt::RegisterTFRTCompiledDialects(registry);
    });

static mlir::TranslateToMLIRRegistration bef_to_mlir_registration(
    "bef-to-mlir", "translate BEF to MLIR",
    [](toolchain::SourceMgr &source_mgr, mlir::MLIRContext *context) {
      mlir::DialectRegistry registry;
      context->appendDialectRegistry(registry);
      return tfrt::BEFToMLIRTranslate(source_mgr, context);
    });
