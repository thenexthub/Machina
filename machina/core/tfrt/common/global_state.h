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
#ifndef MACHINA_CORE_TFRT_COMMON_GLOBAL_STATE_H_
#define MACHINA_CORE_TFRT_COMMON_GLOBAL_STATE_H_

#include "machina/core/framework/resource_mgr.h"
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace machina {
namespace tfrt_global {

class GlobalHostContext {
 public:
  static void Set(::tfrt::HostContext* host_ctx);
  static ::tfrt::HostContext* Get();

 private:
  static ::tfrt::HostContext* host_ctx_;
};

// A global resource manager in TF core framework. It can be used to store
// resources that are per host.
ResourceMgr* GetTFGlobalResourceMgr();

}  // namespace tfrt_global
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_COMMON_GLOBAL_STATE_H_
