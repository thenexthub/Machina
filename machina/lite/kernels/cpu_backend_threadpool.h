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

#ifndef MACHINA_LITE_KERNELS_CPU_BACKEND_THREADPOOL_H_
#define MACHINA_LITE_KERNELS_CPU_BACKEND_THREADPOOL_H_

#include "machina/lite/kernels/cpu_backend_context.h"
#include "machina/lite/kernels/internal/compatibility.h"

#ifdef TFLITE_WITH_RUY
#include "ruy/context.h"  // from @ruy
#include "ruy/thread_pool.h"  // from @ruy
#else
#include "public/gemmlowp.h"
#endif

namespace tflite {
namespace cpu_backend_threadpool {

#ifdef TFLITE_WITH_RUY

using Task = ruy::Task;

template <typename TaskType>
void Execute(int tasks_count, TaskType* tasks,
             CpuBackendContext* cpu_backend_context) {
  TFLITE_DCHECK_LE(tasks_count, cpu_backend_context->max_num_threads());
  cpu_backend_context->ruy_context()->mutable_thread_pool()->Execute(
      tasks_count, tasks);
}

#else  // not TFLITE_WITH_RUY

using Task = gemmlowp::Task;

template <typename TaskType>
void Execute(int tasks_count, TaskType* tasks,
             CpuBackendContext* cpu_backend_context) {
  TFLITE_DCHECK_LE(tasks_count, cpu_backend_context->max_num_threads());
  cpu_backend_context->gemmlowp_context()->workers_pool()->Execute(tasks_count,
                                                                   tasks);
}

#endif

}  // namespace cpu_backend_threadpool
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_CPU_BACKEND_THREADPOOL_H_
