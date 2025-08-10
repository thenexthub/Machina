/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_CORE_PLATFORM_STREAM_EXECUTOR_NO_CUDA_H_
#define MACHINA_CORE_PLATFORM_STREAM_EXECUTOR_NO_CUDA_H_

#include "machina/xla/stream_executor/cuda/cuda_platform_id.h"
#include "machina/xla/stream_executor/device_memory.h"
#include "machina/xla/stream_executor/dnn.h"
#include "machina/xla/stream_executor/event.h"
#include "machina/xla/stream_executor/host/host_platform_id.h"
#include "machina/xla/stream_executor/platform.h"
#include "machina/xla/stream_executor/platform_manager.h"
#include "machina/xla/stream_executor/rocm/rocm_platform_id.h"
#include "machina/xla/stream_executor/scratch_allocator.h"
#include "machina/xla/stream_executor/stream.h"
#include "machina/xla/stream_executor/stream_executor.h"
#include "machina/core/platform/platform.h"
#include "tsl/platform/dso_loader.h"

#endif  // MACHINA_CORE_PLATFORM_STREAM_EXECUTOR_NO_CUDA_H_
