/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#include "machina_serving/servables/machina/thread_pool_factory.h"

#include <memory>
#include <utility>

namespace machina {
namespace serving {

ScopedThreadPools::ScopedThreadPools(
    std::shared_ptr<thread::ThreadPoolInterface> inter_op_thread_pool,
    std::shared_ptr<thread::ThreadPoolInterface> intra_op_thread_pool)
    : inter_op_thread_pool_(std::move(inter_op_thread_pool)),
      intra_op_thread_pool_(std::move(intra_op_thread_pool)) {}

machina::thread::ThreadPoolOptions ScopedThreadPools::get() {
  machina::thread::ThreadPoolOptions options;
  options.inter_op_threadpool = inter_op_thread_pool_.get();
  options.intra_op_threadpool = intra_op_thread_pool_.get();
  return options;
}

}  // namespace serving
}  // namespace machina
