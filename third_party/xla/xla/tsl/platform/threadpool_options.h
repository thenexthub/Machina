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

#ifndef MACHINA_XLATSL_PLATFORM_THREADPOOL_OPTIONS_H_
#define MACHINA_XLATSL_PLATFORM_THREADPOOL_OPTIONS_H_

#include "machina/xla/tsl/platform/threadpool_interface.h"

namespace tsl {
namespace thread {

struct ThreadPoolOptions {
  // If not null, use this threadpool to schedule inter-op operation
  thread::ThreadPoolInterface* inter_op_threadpool = nullptr;

  // If not null, use this threadpool to schedule intra-op operation
  thread::ThreadPoolInterface* intra_op_threadpool = nullptr;
};

}  // namespace thread
}  // namespace tsl

#endif  // MACHINA_XLATSL_PLATFORM_THREADPOOL_OPTIONS_H_
