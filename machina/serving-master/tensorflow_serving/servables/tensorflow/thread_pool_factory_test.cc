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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/core/platform/env.h"
#include "machina/core/platform/threadpool.h"
#include "machina/core/platform/threadpool_options.h"
#include "machina_serving/test_util/test_util.h"

namespace machina {
namespace serving {
namespace {

TEST(ScopedThreadPools, DefaultCtor) {
  ScopedThreadPools thread_pools;
  EXPECT_EQ(nullptr, thread_pools.get().inter_op_threadpool);
  EXPECT_EQ(nullptr, thread_pools.get().intra_op_threadpool);
}

TEST(ScopedThreadPools, NonDefaultCtor) {
  auto inter_op_thread_pool =
      std::make_shared<test_util::CountingThreadPool>(Env::Default(), "InterOp",
                                                      /*num_threads=*/1);
  auto intra_op_thread_pool =
      std::make_shared<test_util::CountingThreadPool>(Env::Default(), "InterOp",
                                                      /*num_threads=*/1);
  ScopedThreadPools thread_pools(inter_op_thread_pool, intra_op_thread_pool);
  EXPECT_EQ(inter_op_thread_pool.get(), thread_pools.get().inter_op_threadpool);
  EXPECT_EQ(intra_op_thread_pool.get(), thread_pools.get().intra_op_threadpool);
}

}  // namespace
}  // namespace serving
}  // namespace machina
