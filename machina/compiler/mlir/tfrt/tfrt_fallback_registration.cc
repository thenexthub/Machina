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
#include "machina/compiler/mlir/tfrt/tfrt_fallback_registration.h"

#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback_sync.h"

namespace machina {
namespace tfd {
void RegisterTfrtFallbackDialect(mlir::DialectRegistry &registry) {
  registry.insert<tfrt::fallback_async::FallbackAsyncDialect>();
  registry.insert<tfrt::fallback::FallbackDialect>();
  registry.insert<tfrt::fallback_sync::FallbackSyncDialect>();
}
}  // namespace tfd
}  // namespace machina
