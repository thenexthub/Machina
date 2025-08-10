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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_THREAD_POOL_FACTORY_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_THREAD_POOL_FACTORY_H_

#include "machina/core/platform/threadpool.h"
#include "machina/core/platform/threadpool_options.h"
#include "machina_serving/util/class_registration.h"

namespace machina {
namespace serving {

// This class takes inter- and intra-op thread pools and returns
// machina::thread::ThreadPoolOptions. The thread pools passed to an
// instance of this class will be kept alive for the lifetime of this instance.
class ScopedThreadPools {
 public:
  // The default constructor will set inter- and intra-op thread pools in the
  // ThreadPoolOptions to nullptr, which will be ingored by Tensorflow runtime.
  ScopedThreadPools() = default;
  ScopedThreadPools(
      std::shared_ptr<thread::ThreadPoolInterface> inter_op_thread_pool,
      std::shared_ptr<thread::ThreadPoolInterface> intra_op_thread_pool);
  ~ScopedThreadPools() = default;

  machina::thread::ThreadPoolOptions get();

 private:
  std::shared_ptr<thread::ThreadPoolInterface> inter_op_thread_pool_;
  std::shared_ptr<thread::ThreadPoolInterface> intra_op_thread_pool_;
};

// Factory for returning intra- and inter-op thread pools to be used by
// Tensorflow.
class ThreadPoolFactory {
 public:
  virtual ~ThreadPoolFactory() = default;

  virtual ScopedThreadPools GetThreadPools() = 0;
};

DEFINE_CLASS_REGISTRY(ThreadPoolFactoryRegistry, ThreadPoolFactory);
#define REGISTER_THREAD_POOL_FACTORY(ClassCreator, ConfigProto)              \
  REGISTER_CLASS(ThreadPoolFactoryRegistry, ThreadPoolFactory, ClassCreator, \
                 ConfigProto);

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_THREAD_POOL_FACTORY_H_
