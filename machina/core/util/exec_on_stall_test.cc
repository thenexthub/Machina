/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include "machina/core/util/exec_on_stall.h"

#include <functional>
#include <memory>
#include <utility>

#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

struct Chunk {
  std::unique_ptr<ExecuteOnStall> stall_closure;
};

Chunk* NewChunk(int stall_seconds, std::function<void()> f) {
  Chunk* c = new Chunk;
  c->stall_closure =
      std::make_unique<ExecuteOnStall>(stall_seconds, std::move(f));
  return c;
}

TEST(ExecuteOnStallTest, BothWays) {
  mutex mu;
  bool a_triggered(false);
  bool b_triggered(false);
  Chunk* a = NewChunk(1, [&mu, &a_triggered]() {
    mutex_lock l(mu);
    a_triggered = true;
  });
  Chunk* b = NewChunk(1, [&mu, &b_triggered]() {
    mutex_lock l(mu);
    b_triggered = true;
  });
  delete a;
  Env::Default()->SleepForMicroseconds(2000000);
  {
    mutex_lock l(mu);
    EXPECT_FALSE(a_triggered);
    EXPECT_TRUE(b_triggered);
  }
  delete b;
}

}  // namespace
}  // namespace machina
