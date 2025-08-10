/* Copyright 2020 The OpenXLA Authors.

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

#ifndef MACHINA_XLASTREAM_EXECUTOR_TPU_C_API_DEFN_H_
#define MACHINA_XLASTREAM_EXECUTOR_TPU_C_API_DEFN_H_

#include <memory>

#include "machina/xla/stream_executor/event.h"
#include "machina/xla/stream_executor/stream.h"
#include "machina/xla/stream_executor/stream_executor.h"

// Definitions for XLA API data structures. Any underlying C++ data structures
// are implementation details and should only be used from within the stream
// executor implementation.

namespace stream_executor {
class Platform;
class StreamExecutor;
}  // namespace stream_executor

struct SE_Platform {
  stream_executor::Platform* platform;
};

struct SE_StreamExecutor {
  stream_executor::StreamExecutor* executor;
};

struct SE_Stream {
  explicit SE_Stream(stream_executor::StreamExecutor* parent) {}
  std::unique_ptr<stream_executor::Stream> stream;
};

struct SE_Event {
  explicit SE_Event(stream_executor::StreamExecutor* parent) {}
  std::unique_ptr<stream_executor::Event> event;
};

// Ignored -- these are just used to enforce the interface types
struct MACHINA_XLATransferManager {};
struct MACHINA_XLAComputationPlacer {};
struct SE_TpuTopology {};
struct SE_TpuTopology_Core {};

#endif  // MACHINA_XLASTREAM_EXECUTOR_TPU_C_API_DEFN_H_
