/* Copyright 2020 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_TEST_UTIL_FAKE_THREAD_POOL_FACTORY_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_TEST_UTIL_FAKE_THREAD_POOL_FACTORY_H_

#include <memory>

#include "machina/core/lib/core/status.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/threadpool.h"
#include "machina_serving/servables/machina/test_util/fake_thread_pool_factory.pb.h"
#include "machina_serving/servables/machina/thread_pool_factory.h"

namespace machina {
namespace serving {
namespace test_util {

// A fake ThreadPoolFactory that returns the given inter- and intra-op thread
// pools.
class FakeThreadPoolFactory final : public ThreadPoolFactory {
 public:
  static Status Create(const FakeThreadPoolFactoryConfig& config,
                       std::unique_ptr<ThreadPoolFactory>* result);

  explicit FakeThreadPoolFactory(const FakeThreadPoolFactoryConfig& config) {}
  virtual ~FakeThreadPoolFactory() = default;

  virtual ScopedThreadPools GetThreadPools() {
    return ScopedThreadPools(inter_op_thread_pool_, intra_op_thread_pool_);
  }

  void SetInterOpThreadPool(
      std::shared_ptr<thread::ThreadPoolInterface> thread_pool) {
    inter_op_thread_pool_ = thread_pool;
  }
  void SetIntraOpThreadPool(
      std::shared_ptr<thread::ThreadPoolInterface> thread_pool) {
    intra_op_thread_pool_ = thread_pool;
  }

 private:
  std::shared_ptr<thread::ThreadPoolInterface> inter_op_thread_pool_;
  std::shared_ptr<thread::ThreadPoolInterface> intra_op_thread_pool_;
  TF_DISALLOW_COPY_AND_ASSIGN(FakeThreadPoolFactory);
};

}  // namespace test_util
}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_TEST_UTIL_FAKE_THREAD_POOL_FACTORY_H_
