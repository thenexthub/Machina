/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#include "tsl/platform/random.h"

#include <memory>
#include <random>

#include "machina/xla/tsl/platform/types.h"
#include "tsl/platform/mutex.h"

namespace tsl {
namespace random {

namespace {
std::mt19937_64* InitRngWithRandomSeed() {
  std::random_device device("/dev/urandom");
  return new std::mt19937_64(device());
}
std::mt19937_64 InitRngWithDefaultSeed() { return std::mt19937_64(); }

}  // anonymous namespace

uint64 New64() {
  static std::mt19937_64* rng = InitRngWithRandomSeed();
  static mutex mu(LINKER_INITIALIZED);
  mutex_lock l(mu);
  return (*rng)();
}

uint64 ThreadLocalNew64() {
  static thread_local std::unique_ptr<std::mt19937_64> rng =
      std::unique_ptr<std::mt19937_64>(InitRngWithRandomSeed());
  return (*rng)();
}

uint64 New64DefaultSeed() {
  static std::mt19937_64 rng = InitRngWithDefaultSeed();
  static mutex mu(LINKER_INITIALIZED);
  mutex_lock l(mu);
  return rng();
}

}  // namespace random
}  // namespace tsl
