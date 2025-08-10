/* Copyright 2023 The OpenXLA Authors.

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

#include "machina/xla/ffi/ffi_api.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "machina/xla/executable_run_options.h"
#include "machina/xla/ffi/api/api.h"
#include "machina/xla/ffi/api/c_api.h"
#include "machina/xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "machina/xla/ffi/call_frame.h"
#include "machina/xla/ffi/execution_context.h"
#include "machina/xla/ffi/execution_state.h"
#include "machina/xla/ffi/type_id_registry.h"
#include "machina/xla/hlo/ir/hlo_computation.h"
#include "machina/xla/service/platform_util.h"
#include "machina/xla/stream_executor/device_memory.h"
#include "machina/xla/stream_executor/device_memory_allocator.h"
#include "machina/xla/stream_executor/stream.h"
#include "machina/xla/tsl/concurrency/async_value.h"
#include "machina/xla/tsl/concurrency/async_value_ref.h"
#include "machina/xla/tsl/concurrency/chain.h"
#include "machina/xla/tsl/concurrency/ref_count.h"
#include "machina/xla/tsl/platform/logging.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/xla/util.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

//===----------------------------------------------------------------------===//
// XLA FFI C structs definition
//===----------------------------------------------------------------------===//

struct MACHINA_MACHINA_XLA_FFI_Error {
  absl::Status status;
};

struct MACHINA_MACHINA_XLA_FFI_Future {
  tsl::AsyncValueRef<tsl::Chain> async_value;
};

struct MACHINA_MACHINA_XLA_FFI_ExecutionContext {
  struct CpuContext {
    const Eigen::ThreadPoolDevice* intra_op_thread_pool = nullptr;
  };

  struct GpuContext {
    stream_executor::Stream* stream = nullptr;
    stream_executor::DeviceMemoryAllocator* allocator = nullptr;
  };

  using BackendContext = std::variant<std::monostate, CpuContext, GpuContext>;

  xla::RunId run_id = {};
  int32_t device_ordinal = -1;
  BackendContext backend_context = {};

  const xla::HloComputation* called_computation = nullptr;
  const xla::ffi::ExecutionContext* execution_context = nullptr;
  xla::ffi::ExecutionState* execution_state = nullptr;
};

//===----------------------------------------------------------------------===//

namespace xla::ffi {

bool IsCommandBufferCompatible(MACHINA_MACHINA_XLA_FFI_Handler_Traits traits) {
  return traits & MACHINA_MACHINA_XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
}

static MACHINA_MACHINA_XLA_FFI_ExecutionContext CreateExecutionContext(
    const CallOptions& options) {
  using BackendContext = MACHINA_MACHINA_XLA_FFI_ExecutionContext::BackendContext;

  // Converts CallOptions to corresponding backend context.
  struct BackendVisitor {
    BackendContext operator()(const std::monostate&) const {
      return std::monostate{};
    }

    BackendContext operator()(const CallOptions::CpuOptions& options) const {
      return MACHINA_MACHINA_XLA_FFI_ExecutionContext::CpuContext{options.intra_op_thread_pool};
    }

    BackendContext operator()(const CallOptions::GpuOptions& options) const {
      return MACHINA_MACHINA_XLA_FFI_ExecutionContext::GpuContext{options.stream,
                                                  options.allocator};
    }
  };

  return MACHINA_MACHINA_XLA_FFI_ExecutionContext{
      options.run_id,
      options.device_ordinal,
      std::visit(BackendVisitor{}, options.backend_options),
      options.called_computation,
      internal::ScopedExecutionContext::GetCallExecutionContext(options),
      options.execution_state,
  };
}

//===----------------------------------------------------------------------===//
// Calling XLA FFI handlers
//===----------------------------------------------------------------------===//

absl::Status TakeStatus(MACHINA_MACHINA_XLA_FFI_Error* error) {
  if (ABSL_PREDICT_TRUE(error == nullptr)) return absl::OkStatus();
  absl::Status status = std::move(error->status);
  delete error;
  return status;
}

tsl::AsyncValueRef<tsl::Chain> TakeFuture(MACHINA_MACHINA_XLA_FFI_Future* future) {
  // Non-reference-counted async value ref for synchronous FFI handlers.
  static tsl::AsyncValueOwningRef<tsl::Chain>* chain = [] {
    auto* storage = new tsl::internal::AsyncValueStorage<tsl::Chain>();
    return new tsl::AsyncValueOwningRef<tsl::Chain>(
        tsl::MakeAvailableAsyncValueRef<tsl::Chain>(*storage));
  }();

  if (ABSL_PREDICT_TRUE(future == nullptr)) return chain->AsRef();

  // If the future is already completed, immediately return the underlying async
  // value and delete the MACHINA_MACHINA_XLA_FFI_Future.
  if (ABSL_PREDICT_TRUE(future->async_value.IsAvailable())) {
    tsl::AsyncValueRef<tsl::Chain> async_value = std::move(future->async_value);
    delete future;
    return async_value;
  }

  // If the future is not completed, return a copy of the underlying async value
  // and keep MACHINA_MACHINA_XLA_FFI_Future alive until it is completed.
  tsl::AsyncValueRef<tsl::Chain> async_value = future->async_value;
  async_value.AndThen([future] { delete future; });
  return async_value;
}

template <typename Handler>
static absl::StatusOr<MACHINA_MACHINA_XLA_FFI_Future*> Call(Handler& handler,
                                            CallFrame& call_frame,
                                            const CallOptions& options,
                                            ExecutionStage stage) {
  MACHINA_MACHINA_XLA_FFI_ExecutionContext ctx = CreateExecutionContext(options);
  MACHINA_MACHINA_XLA_FFI_CallFrame ffi_call_frame = call_frame.Build(
      GetXlaFfiApi(), &ctx, static_cast<MACHINA_MACHINA_XLA_FFI_ExecutionStage>(stage));

  MACHINA_MACHINA_XLA_FFI_Error* error = nullptr;

  // FFI handlers might be defined in external libraries and use exceptions, so
  // take extra care to catch them and convert to a status.
  try {
    if constexpr (std::is_same_v<Handler, Ffi>) {
      error = handler.Call(&ffi_call_frame);
    } else if constexpr (std::is_same_v<Handler, MACHINA_MACHINA_XLA_FFI_Handler*>) {
      error = (*handler)(&ffi_call_frame);
    } else {
      static_assert(sizeof(Handler) == 0, "Unsupported handler type");
    }
  } catch (std::exception& e) {
    return Unknown("XLA FFI call failed: %s", e.what());
  }

  // If FFI handler returned synchronous error, it must not launch any
  // asynchronous work that can also return an error.
  if (error != nullptr) {
    DCHECK_EQ(ffi_call_frame.future, nullptr)
        << "Error must not be used together with a future";
    return TakeStatus(error);
  }

  return ffi_call_frame.future;
}

static absl::Status BlockUntilReady(MACHINA_MACHINA_XLA_FFI_Future* future) {
  if (ABSL_PREDICT_TRUE(future == nullptr)) return absl::OkStatus();

  tsl::AsyncValueRef<tsl::Chain> av = TakeFuture(future);
  tsl::BlockUntilReady(av);
  return ABSL_PREDICT_FALSE(av.IsError()) ? av.GetError() : absl::OkStatus();
}

absl::Status Call(Ffi& handler, CallFrame& call_frame,
                  const CallOptions& options, ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(MACHINA_MACHINA_XLA_FFI_Future * future,
                      Call<Ffi>(handler, call_frame, options, stage));
  return BlockUntilReady(future);
}

absl::Status Call(MACHINA_MACHINA_XLA_FFI_Handler* handler, CallFrame& call_frame,
                  const CallOptions& options, MACHINA_MACHINA_XLA_FFI_ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(
      MACHINA_MACHINA_XLA_FFI_Future * future,
      Call<MACHINA_MACHINA_XLA_FFI_Handler*>(handler, call_frame, options,
                             static_cast<ExecutionStage>(stage)));
  return BlockUntilReady(future);
}

tsl::AsyncValueRef<tsl::Chain> CallAsync(Ffi& handler, CallFrame& call_frame,
                                         const CallOptions& options,
                                         ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(MACHINA_MACHINA_XLA_FFI_Future * future,
                      Call<Ffi>(handler, call_frame, options, stage));
  return TakeFuture(future);
}

tsl::AsyncValueRef<tsl::Chain> CallAsync(MACHINA_MACHINA_XLA_FFI_Handler* handler,
                                         CallFrame& call_frame,
                                         const CallOptions& options,
                                         MACHINA_MACHINA_XLA_FFI_ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(
      MACHINA_MACHINA_XLA_FFI_Future * future,
      Call<MACHINA_MACHINA_XLA_FFI_Handler*>(handler, call_frame, options,
                             static_cast<ExecutionStage>(stage)));
  return TakeFuture(future);
}

static MACHINA_MACHINA_XLA_FFI_Metadata BuildMetadata() {
  return MACHINA_MACHINA_XLA_FFI_Metadata{MACHINA_MACHINA_XLA_FFI_Metadata_STRUCT_SIZE,
                          MACHINA_MACHINA_XLA_FFI_Api_Version{MACHINA_MACHINA_XLA_FFI_Api_Version_STRUCT_SIZE}};
}

static MACHINA_MACHINA_XLA_FFI_Metadata_Extension BuildMetadataExtension(
    MACHINA_MACHINA_XLA_FFI_Metadata* metadata) {
  return MACHINA_MACHINA_XLA_FFI_Metadata_Extension{
      MACHINA_MACHINA_XLA_FFI_Extension_Base{MACHINA_MACHINA_XLA_FFI_Metadata_Extension_STRUCT_SIZE,
                             MACHINA_MACHINA_XLA_FFI_Extension_Metadata},
      metadata};
}

static MACHINA_MACHINA_XLA_FFI_CallFrame BuildMetadataCallFrame(
    MACHINA_MACHINA_XLA_FFI_Metadata_Extension* extension) {
  return MACHINA_MACHINA_XLA_FFI_CallFrame{
      MACHINA_MACHINA_XLA_FFI_CallFrame_STRUCT_SIZE,
      &extension->extension_base,
      /*api=*/nullptr,
      /*context=*/nullptr,
      /*stage=*/MACHINA_MACHINA_XLA_FFI_ExecutionStage_EXECUTE,
      /*args=*/MACHINA_MACHINA_XLA_FFI_Args{MACHINA_MACHINA_XLA_FFI_Args_STRUCT_SIZE},
      /*rets=*/MACHINA_MACHINA_XLA_FFI_Rets{MACHINA_MACHINA_XLA_FFI_Rets_STRUCT_SIZE},
      /*attrs=*/MACHINA_MACHINA_XLA_FFI_Attrs{MACHINA_MACHINA_XLA_FFI_Attrs_STRUCT_SIZE},
  };
}

absl::StatusOr<MACHINA_MACHINA_XLA_FFI_Metadata> GetMetadata(Ffi& handler) {
  MACHINA_MACHINA_XLA_FFI_Metadata metadata = BuildMetadata();
  MACHINA_MACHINA_XLA_FFI_Metadata_Extension extension = BuildMetadataExtension(&metadata);
  MACHINA_MACHINA_XLA_FFI_CallFrame call_frame = BuildMetadataCallFrame(&extension);
  MACHINA_MACHINA_XLA_FFI_Error* error = nullptr;
  try {
    error = handler.Call(&call_frame);
  } catch (std::exception& e) {
    return Unknown("Fetching XLA FFI metadata failed: %s", e.what());
  }
  if (error != nullptr) {
    return TakeStatus(error);
  }
  return metadata;
}

absl::StatusOr<MACHINA_MACHINA_XLA_FFI_Metadata> GetMetadata(MACHINA_MACHINA_XLA_FFI_Handler* handler) {
  MACHINA_MACHINA_XLA_FFI_Metadata metadata = BuildMetadata();
  MACHINA_MACHINA_XLA_FFI_Metadata_Extension extension = BuildMetadataExtension(&metadata);
  MACHINA_MACHINA_XLA_FFI_CallFrame call_frame = BuildMetadataCallFrame(&extension);
  MACHINA_MACHINA_XLA_FFI_Error* error = nullptr;
  try {
    error = (*handler)(&call_frame);
  } catch (std::exception& e) {
    return Unknown("Fetching XLA FFI metadata failed: %s", e.what());
  }
  if (error != nullptr) {
    return TakeStatus(error);
  }
  return metadata;
}

namespace internal {
static thread_local const ExecutionContext* scoped_execution_context = nullptr;

ScopedExecutionContext::ScopedExecutionContext(const ExecutionContext* context)
    : recover_(scoped_execution_context) {
  scoped_execution_context = context;
}

ScopedExecutionContext::~ScopedExecutionContext() {
  scoped_execution_context = recover_;
}

const ExecutionContext* ScopedExecutionContext::GetCallExecutionContext(
    const CallOptions& options) {
  if (scoped_execution_context != nullptr) {
    return scoped_execution_context;
  }
  return options.execution_context;
}
}  // namespace internal

//===----------------------------------------------------------------------===//
// XLA FFI registry
//===----------------------------------------------------------------------===//

using HandlerKey = std::pair<std::string, std::string>;
using HandlerRegistry = absl::flat_hash_map<HandlerKey, HandlerRegistration>;

static HandlerKey MakeHandlerKey(absl::string_view name,
                                 absl::string_view platform) {
  return std::make_pair(std::string(name), absl::AsciiStrToLower(platform));
}

static HandlerRegistry& GetHandlerRegistry() {
  static auto* const registry = new HandlerRegistry();
  return *registry;
}

static std::vector<std::string> GetHandlerStages(
    const MACHINA_MACHINA_XLA_FFI_Handler_Bundle& bundle) {
  std::vector<std::string> stages;
  if (bundle.instantiate != nullptr) stages.push_back("instantiate");
  if (bundle.prepare != nullptr) stages.push_back("prepare");
  if (bundle.initialize != nullptr) stages.push_back("initialize");
  if (bundle.execute != nullptr) stages.push_back("execute");
  return stages;
}

static absl::Status RegisterHandler(absl::string_view name,
                                    absl::string_view platform,
                                    MACHINA_MACHINA_XLA_FFI_Handler_Bundle bundle,
                                    MACHINA_MACHINA_XLA_FFI_Handler_Traits traits) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  if (bundle.execute == nullptr) {
    return InvalidArgument(
        "FFI handler for %s on a platform %s must provide an execute "
        "implementation",
        name, platform);
  }

  // Check the API versions.
  TF_ASSIGN_OR_RETURN(auto metadata, GetMetadata(bundle.execute));
  const MACHINA_MACHINA_XLA_FFI_Api_Version& api_version = metadata.api_version;
  if (api_version.major_version != MACHINA_MACHINA_XLA_FFI_API_MAJOR ||
      api_version.minor_version != MACHINA_MACHINA_XLA_FFI_API_MINOR) {
    return InvalidArgument(
        "FFI handler registration for %s on platform %s (canonical %s) failed "
        "because the handler's API version (%d.%d) is incompatible with the "
        "framework's API version (%d.%d)",
        name, platform, canonical_platform, api_version.major_version,
        api_version.minor_version, MACHINA_MACHINA_XLA_FFI_API_MAJOR, MACHINA_MACHINA_XLA_FFI_API_MINOR);
  }

  // Incorporate handler traits.
  traits |= metadata.traits;

  VLOG(2) << absl::StreamFormat(
      "Register XLA FFI handler for '%s'; platform=%s (canonical=%s), "
      "stages=[%s], command_buffer_compatible=%v",
      name, platform, canonical_platform,
      absl::StrJoin(GetHandlerStages(bundle), ", "),
      IsCommandBufferCompatible(traits));

  auto emplaced =
      GetHandlerRegistry().try_emplace(MakeHandlerKey(name, canonical_platform),
                                       HandlerRegistration{bundle, traits});
  if (!emplaced.second) {
    auto existing = emplaced.first->second;
    if (existing.traits != traits) {
      return InvalidArgument(
          "Duplicate FFI handler registration for %s on platform %s "
          "(canonical %s) with different traits",
          name, platform, canonical_platform);
    }
    if (existing.bundle.prepare != bundle.prepare ||
        existing.bundle.initialize != bundle.initialize ||
        existing.bundle.execute != bundle.execute) {
      return InvalidArgument(
          "Duplicate FFI handler registration for %s on platform %s "
          "(canonical %s) with different bundle addresses",
          name, platform, canonical_platform);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<HandlerRegistration> FindHandler(absl::string_view name,
                                                absl::string_view platform) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  auto it = GetHandlerRegistry().find(MakeHandlerKey(name, canonical_platform));
  if (it == GetHandlerRegistry().end()) {
    return NotFound(
        "No FFI handler registered for %s on a platform %s (canonical %s)",
        name, platform, canonical_platform);
  }
  return it->second;
}

absl::StatusOr<absl::flat_hash_map<std::string, HandlerRegistration>>
StaticRegisteredHandlers(absl::string_view platform) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  absl::flat_hash_map<std::string, HandlerRegistration> calls;
  for (const auto& [metadata, handler] : GetHandlerRegistry()) {
    if (canonical_platform == metadata.second) {
      calls[metadata.first] = handler;
    }
  }

  return calls;
}

//===----------------------------------------------------------------------===//
// XLA FFI Api Implementation
//===----------------------------------------------------------------------===//

static std::string StructSizeErrorMsg(absl::string_view struct_name,
                                      size_t expected, size_t actual) {
  return absl::StrCat("Unexpected ", struct_name, " size: expected ", expected,
                      ", got ", actual, ". Check installed software versions. ",
                      "The framework XLA FFI API version is ",
                      MACHINA_MACHINA_XLA_FFI_API_MAJOR, ".", MACHINA_MACHINA_XLA_FFI_API_MINOR, ".");
}

static absl::Status ActualStructSizeIsGreaterOrEqual(
    absl::string_view struct_name, size_t expected, size_t actual) {
  if (actual < expected) {
    return InvalidArgument("%s",
                           StructSizeErrorMsg(struct_name, expected, actual));
  }
  if (actual > expected) {
    VLOG(2) << StructSizeErrorMsg(struct_name, expected, actual);
  }
  return absl::OkStatus();
}

static absl::StatusCode ToStatusCode(MACHINA_MACHINA_XLA_FFI_Error_Code errc) {
  switch (errc) {
    case MACHINA_MACHINA_XLA_FFI_Error_Code_OK:
      return absl::StatusCode::kOk;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_CANCELLED:
      return absl::StatusCode::kCancelled;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_UNKNOWN:
      return absl::StatusCode::kUnknown;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_INVALID_ARGUMENT:
      return absl::StatusCode::kInvalidArgument;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_DEADLINE_EXCEEDED:
      return absl::StatusCode::kDeadlineExceeded;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_NOT_FOUND:
      return absl::StatusCode::kNotFound;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_ALREADY_EXISTS:
      return absl::StatusCode::kAlreadyExists;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_PERMISSION_DENIED:
      return absl::StatusCode::kPermissionDenied;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_RESOURCE_EXHAUSTED:
      return absl::StatusCode::kResourceExhausted;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_FAILED_PRECONDITION:
      return absl::StatusCode::kFailedPrecondition;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_ABORTED:
      return absl::StatusCode::kAborted;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_OUT_OF_RANGE:
      return absl::StatusCode::kOutOfRange;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_UNIMPLEMENTED:
      return absl::StatusCode::kUnimplemented;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_INTERNAL:
      return absl::StatusCode::kInternal;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_UNAVAILABLE:
      return absl::StatusCode::kUnavailable;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_DATA_LOSS:
      return absl::StatusCode::kDataLoss;
    case MACHINA_MACHINA_XLA_FFI_Error_Code_UNAUTHENTICATED:
      return absl::StatusCode::kUnauthenticated;
  }
}

#define MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(expr)                                   \
  do {                                                                  \
    absl::Status _status = (expr);                                      \
    if (!_status.ok()) {                                                \
      MACHINA_MACHINA_XLA_FFI_Error* _c_status = new MACHINA_MACHINA_XLA_FFI_Error{std::move(_status)}; \
      return _c_status;                                                 \
    }                                                                   \
  } while (false)

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Error_Create(MACHINA_MACHINA_XLA_FFI_Error_Create_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_Error_Create", MACHINA_MACHINA_XLA_FFI_Error_Create_Args_STRUCT_SIZE,
      args->struct_size));

  return new MACHINA_MACHINA_XLA_FFI_Error{
      absl::Status(ToStatusCode(args->errc), args->message)};
}

static void MACHINA_MACHINA_XLA_FFI_Error_GetMessage(MACHINA_MACHINA_XLA_FFI_Error_GetMessage_Args* args) {
  absl::Status struct_size_check = ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_Error_GetMessage", MACHINA_MACHINA_XLA_FFI_Error_GetMessage_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.message();
  }
  // absl::Status owns error message in a std::string which guarantees that
  // we'll get a null terminated string.
  args->message = args->error->status.message().data();
}

static void MACHINA_MACHINA_XLA_FFI_Error_Destroy(MACHINA_MACHINA_XLA_FFI_Error_Destroy_Args* args) {
  absl::Status struct_size_check = ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_Error_Destroy", MACHINA_MACHINA_XLA_FFI_Error_Destroy_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.message();
  }
  delete args->error;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Future_Create(MACHINA_MACHINA_XLA_FFI_Future_Create_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_Future_Create", MACHINA_MACHINA_XLA_FFI_Future_Create_Args_STRUCT_SIZE,
      args->struct_size));
  args->future =
      new MACHINA_MACHINA_XLA_FFI_Future{tsl::MakeConstructedAsyncValueRef<tsl::Chain>()};
  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Future_SetAvailable(
    MACHINA_MACHINA_XLA_FFI_Future_SetAvailable_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_Future_SetAvailable",
      MACHINA_MACHINA_XLA_FFI_Future_SetAvailable_Args_STRUCT_SIZE, args->struct_size));
  args->future->async_value.SetStateConcrete();
  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Future_SetError(
    MACHINA_MACHINA_XLA_FFI_Future_SetError_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_Future_SetError", MACHINA_MACHINA_XLA_FFI_Future_SetError_Args_STRUCT_SIZE,
      args->struct_size));

  if (args->error == nullptr || args->error->status.ok()) {
    return new MACHINA_MACHINA_XLA_FFI_Error{InvalidArgument("Error must not be null or OK")};
  }

  absl::Status error = TakeStatus(args->error);
  args->future->async_value.SetError(std::move(error));

  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Handler_Register(
    MACHINA_MACHINA_XLA_FFI_Handler_Register_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_Handler_Register", MACHINA_MACHINA_XLA_FFI_Handler_Register_Args_STRUCT_SIZE,
      args->struct_size));

  if (auto status = RegisterHandler(
          absl::string_view(args->name.ptr, args->name.len),
          absl::string_view(args->platform.ptr, args->platform.len),
          args->bundle, args->traits);
      !status.ok()) {
    return new MACHINA_MACHINA_XLA_FFI_Error{std::move(status)};
  }
  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Stream_Get(MACHINA_MACHINA_XLA_FFI_Stream_Get_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_Stream_Get", MACHINA_MACHINA_XLA_FFI_Stream_Get_Args_STRUCT_SIZE,
      args->struct_size));

  auto* gpu = std::get_if<MACHINA_MACHINA_XLA_FFI_ExecutionContext::GpuContext>(
      &args->ctx->backend_context);

  if (ABSL_PREDICT_FALSE(gpu == nullptr)) {
    return new MACHINA_MACHINA_XLA_FFI_Error{
        Unimplemented("XLA FFI GPU context is not available")};
  }

  if (ABSL_PREDICT_FALSE(gpu->stream == nullptr)) {
    return new MACHINA_MACHINA_XLA_FFI_Error{
        Unimplemented("XLA FFI GPU stream is not available")};
  }

  auto handle = gpu->stream->platform_specific_handle();
  args->stream = handle.stream;

  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_RunId_Get(MACHINA_MACHINA_XLA_FFI_RunId_Get_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_RunId_Get", MACHINA_MACHINA_XLA_FFI_RunId_Get_Args_STRUCT_SIZE,
      args->struct_size));

  args->run_id = args->ctx->run_id.ToInt();

  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_DeviceOrdinal_Get(
    MACHINA_MACHINA_XLA_FFI_DeviceOrdinal_Get_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_DeviceOrdinal_Get", MACHINA_MACHINA_XLA_FFI_DeviceOrdinal_Get_Args_STRUCT_SIZE,
      args->struct_size));
  args->device_ordinal = args->ctx->device_ordinal;
  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_TypeId_Register(
    MACHINA_MACHINA_XLA_FFI_TypeId_Register_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get_Args",
      MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get_Args_STRUCT_SIZE, args->struct_size));

  absl::string_view type_name(args->name.ptr, args->name.len);
  TypeIdRegistry::TypeId type_id(args->type_id->type_id);

  // If type_id is unknown, we are registering a new type and XLA will assign a
  // unique type id to it.
  if (type_id == TypeIdRegistry::kUnknownTypeId) {
    auto assigned_type_id = TypeIdRegistry::AssignExternalTypeId(type_name);
    if (!assigned_type_id.ok()) {
      return new MACHINA_MACHINA_XLA_FFI_Error{std::move(assigned_type_id).status()};
    }

    args->type_id->type_id = assigned_type_id->value();
    return nullptr;
  }

  // If type_id is set, we are relying on the caller-provided unique type id.
  if (auto status = TypeIdRegistry::RegisterExternalTypeId(type_name, type_id);
      !status.ok()) {
    return new MACHINA_MACHINA_XLA_FFI_Error{std::move(status)};
  }

  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get(
    MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get_Args",
      MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get_Args_STRUCT_SIZE, args->struct_size));

  DCHECK(args->ctx->execution_context) << "ExecutionContext must be set";
  auto user_data = args->ctx->execution_context->Lookup(
      TypeIdRegistry::TypeId(args->type_id->type_id));
  if (!user_data.ok()) {
    return new MACHINA_MACHINA_XLA_FFI_Error{std::move(user_data).status()};
  }

  args->data = *user_data;
  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_State_Set(MACHINA_MACHINA_XLA_FFI_State_Set_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_State_Set_Args", MACHINA_MACHINA_XLA_FFI_State_Set_Args_STRUCT_SIZE,
      args->struct_size));

  DCHECK(args->ctx->execution_state) << "ExecutionState must be set";
  absl::Status status = args->ctx->execution_state->Set(
      TypeIdRegistry::TypeId(args->type_id->type_id), args->state,
      [deleter = args->deleter](void* state) { deleter(state); });

  if (!status.ok()) {
    return new MACHINA_MACHINA_XLA_FFI_Error{std::move(status)};
  }

  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_State_Get(MACHINA_MACHINA_XLA_FFI_State_Get_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_State_Get_Args", MACHINA_MACHINA_XLA_FFI_State_Get_Args_STRUCT_SIZE,
      args->struct_size));

  DCHECK(args->ctx->execution_state) << "ExecutionState must be set";
  absl::StatusOr<void*> state = args->ctx->execution_state->Get(
      TypeIdRegistry::TypeId(args->type_id->type_id));
  if (!state.ok()) {
    return new MACHINA_MACHINA_XLA_FFI_Error{std::move(state).status()};
  }

  args->state = *state;
  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_DeviceMemory_Allocate(
    MACHINA_MACHINA_XLA_FFI_DeviceMemory_Allocate_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_DeviceMemory_Allocate_Args",
      MACHINA_MACHINA_XLA_FFI_DeviceMemory_Allocate_Args_STRUCT_SIZE, args->struct_size));

  auto* gpu = std::get_if<MACHINA_MACHINA_XLA_FFI_ExecutionContext::GpuContext>(
      &args->ctx->backend_context);

  // TODO(ezhulenev): Device memory allocation should be supported for all
  // backends, not just GPU, although for CPU it doesn't make much sense, as
  // plain `new` is sufficient.
  if (ABSL_PREDICT_FALSE(gpu == nullptr)) {
    return new MACHINA_MACHINA_XLA_FFI_Error{
        InvalidArgument("XLA FFI GPU context is not available")};
  }

  if (ABSL_PREDICT_FALSE(gpu->allocator == nullptr)) {
    return new MACHINA_MACHINA_XLA_FFI_Error{
        Unimplemented("No device memory allocator available on this platform")};
  }

  // TODO(ezhulenev): We happen to have the same alignment requirement for
  // device memory on CPU and GPU backends, but instead of hardcoding it here
  // we should query it for the platform XLA FFI handler is registered with.
  static constexpr int64_t kMaxAlignment = 16;

  if (!absl::has_single_bit(args->alignment) ||
      args->alignment > kMaxAlignment) {
    return new MACHINA_MACHINA_XLA_FFI_Error{
        InvalidArgument("Unsupported alignment: %d", args->alignment)};
  }

  absl::StatusOr<stream_executor::OwningDeviceMemory> memory =
      gpu->allocator->Allocate(args->ctx->device_ordinal, args->size);
  if (!memory.ok()) {
    return new MACHINA_MACHINA_XLA_FFI_Error{std::move(memory).status()};
  }

  args->data = memory->Release().opaque();
  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_DeviceMemory_Free(
    MACHINA_MACHINA_XLA_FFI_DeviceMemory_Free_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_DeviceMemory_Free_Args",
      MACHINA_MACHINA_XLA_FFI_DeviceMemory_Free_Args_STRUCT_SIZE, args->struct_size));

  auto* gpu = std::get_if<MACHINA_MACHINA_XLA_FFI_ExecutionContext::GpuContext>(
      &args->ctx->backend_context);

  // TODO(ezhulenev): Device memory allocation should be supported for all
  // backends, not just GPU, although for CPU it doesn't make much sense, as
  // plain `new` is sufficient.
  if (ABSL_PREDICT_FALSE(gpu == nullptr)) {
    return new MACHINA_MACHINA_XLA_FFI_Error{
        Unimplemented("XLA FFI GPU context is not available")};
  }

  if (ABSL_PREDICT_FALSE(gpu->allocator == nullptr)) {
    return new MACHINA_MACHINA_XLA_FFI_Error{
        Unimplemented("No device memory allocator available on this platform")};
  }

  absl::Status status = gpu->allocator->Deallocate(
      args->ctx->device_ordinal,
      stream_executor::DeviceMemoryBase(args->data, args->size));
  if (!status.ok()) {
    return new MACHINA_MACHINA_XLA_FFI_Error{std::move(status)};
  }

  return nullptr;
}

static absl::StatusOr<const Eigen::ThreadPoolDevice*> GetIntraOpThreadPool(
    const MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx) {
  auto* cpu =
      std::get_if<MACHINA_MACHINA_XLA_FFI_ExecutionContext::CpuContext>(&ctx->backend_context);

  if (ABSL_PREDICT_FALSE(cpu == nullptr)) {
    return Unimplemented("XLA FFI CPU context is not available");
  }

  if (ABSL_PREDICT_FALSE(cpu->intra_op_thread_pool == nullptr)) {
    return Unimplemented("No intra-op thread pool available on this platform");
  }

  return cpu->intra_op_thread_pool;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_ThreadPool_Schedule(
    MACHINA_MACHINA_XLA_FFI_ThreadPool_Schedule_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_ThreadPool_Schedule_Args",
      MACHINA_MACHINA_XLA_FFI_ThreadPool_Schedule_Args_STRUCT_SIZE, args->struct_size));

  auto intra_op_thread_pool = GetIntraOpThreadPool(args->ctx);
  if (!intra_op_thread_pool.ok()) {
    return new MACHINA_MACHINA_XLA_FFI_Error{std::move(intra_op_thread_pool).status()};
  }

  (*intra_op_thread_pool)
      ->enqueueNoNotification(
          [task = args->task, data = args->data] { (*task)(data); });

  return nullptr;
}

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_ThreadPool_NumThreads(
    MACHINA_MACHINA_XLA_FFI_ThreadPool_NumThreads_Args* args) {
  MACHINA_MACHINA_XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "MACHINA_MACHINA_XLA_FFI_ThreadPool_NumThreads_Args",
      MACHINA_MACHINA_XLA_FFI_ThreadPool_NumThreads_Args_STRUCT_SIZE, args->struct_size));

  auto intra_op_thread_pool = GetIntraOpThreadPool(args->ctx);
  if (!intra_op_thread_pool.ok()) {
    return new MACHINA_MACHINA_XLA_FFI_Error{std::move(intra_op_thread_pool).status()};
  }

  *args->num_threads = (*intra_op_thread_pool)->numThreadsInPool();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// XLA FFI Internal Api Implementation
//===----------------------------------------------------------------------===//

static MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_INTERNAL_Error_Forward(void* status) {
  auto* absl_status = reinterpret_cast<absl::Status*>(status);
  if (ABSL_PREDICT_TRUE(absl_status->ok())) {
    return nullptr;
  }
  return new MACHINA_MACHINA_XLA_FFI_Error{std::move(*absl_status)};
}

static MACHINA_MACHINA_XLA_FFI_Future* MACHINA_MACHINA_XLA_FFI_INTERNAL_Future_Forward(void* async_value) {
  auto* tsl_async_value = reinterpret_cast<tsl::AsyncValue*>(async_value);
  DCHECK(tsl_async_value) << "Async value must not be null";

  return new MACHINA_MACHINA_XLA_FFI_Future{
      tsl::AsyncValueRef<tsl::Chain>(tsl::TakeRef(tsl_async_value))};
}

static void* MACHINA_MACHINA_XLA_FFI_INTERNAL_Stream_Get(MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx) {
  if (auto* gpu = std::get_if<MACHINA_MACHINA_XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    return gpu->stream;
  }

  return new MACHINA_MACHINA_XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

static int32_t MACHINA_MACHINA_XLA_FFI_INTERNAL_DeviceOrdinal_Get(
    MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx) {
  return ctx->device_ordinal;
}

static int64_t MACHINA_MACHINA_XLA_FFI_INTERNAL_RunId_Get(MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx) {
  return ctx->run_id.ToInt();
}

static void* MACHINA_MACHINA_XLA_FFI_INTERNAL_DeviceMemoryAllocator_Get(
    MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx) {
  if (auto* gpu = std::get_if<MACHINA_MACHINA_XLA_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    return gpu->allocator;
  }

  return new MACHINA_MACHINA_XLA_FFI_Error{
      InvalidArgument("XLA FFI GPU context is not available")};
}

static void* MACHINA_MACHINA_XLA_FFI_INTERNAL_CalledComputation_Get(
    MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx) {
  return const_cast<HloComputation*>(ctx->called_computation);
}

static void* MACHINA_MACHINA_XLA_FFI_INTERNAL_ExecutionContext_Get(
    MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx) {
  return const_cast<ffi::ExecutionContext*>(ctx->execution_context);
}

static void* MACHINA_MACHINA_XLA_FFI_INTERNAL_ExecutionState_Get(
    MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx) {
  return const_cast<ffi::ExecutionState*>(ctx->execution_state);
}

void* MACHINA_MACHINA_XLA_FFI_INTERNAL_IntraOpThreadPool_Get(MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx) {
  if (auto* cpu = std::get_if<MACHINA_MACHINA_XLA_FFI_ExecutionContext::CpuContext>(
          &ctx->backend_context)) {
    return const_cast<Eigen::ThreadPoolDevice*>(cpu->intra_op_thread_pool);
  }

  return new MACHINA_MACHINA_XLA_FFI_Error{
      InvalidArgument("XLA FFI CPU context is not available")};
}

//===----------------------------------------------------------------------===//
// XLA FFI Api access
//===----------------------------------------------------------------------===//

extern "C" const MACHINA_MACHINA_XLA_FFI_Api* MACHINA_MACHINA_XLA_FFI_GetApi() { return GetXlaFfiApi(); }

static MACHINA_MACHINA_XLA_FFI_InternalApi internal_api = {
    MACHINA_MACHINA_XLA_FFI_INTERNAL_Error_Forward,
    MACHINA_MACHINA_XLA_FFI_INTERNAL_Future_Forward,
    MACHINA_MACHINA_XLA_FFI_INTERNAL_Stream_Get,
    MACHINA_MACHINA_XLA_FFI_INTERNAL_DeviceOrdinal_Get,
    MACHINA_MACHINA_XLA_FFI_INTERNAL_RunId_Get,
    MACHINA_MACHINA_XLA_FFI_INTERNAL_DeviceMemoryAllocator_Get,
    MACHINA_MACHINA_XLA_FFI_INTERNAL_CalledComputation_Get,
    MACHINA_MACHINA_XLA_FFI_INTERNAL_ExecutionContext_Get,
    MACHINA_MACHINA_XLA_FFI_INTERNAL_ExecutionState_Get,
    MACHINA_MACHINA_XLA_FFI_INTERNAL_IntraOpThreadPool_Get,
};

static MACHINA_MACHINA_XLA_FFI_Api api = {
    MACHINA_MACHINA_XLA_FFI_Api_STRUCT_SIZE,
    /*extension_start=*/nullptr,

    MACHINA_MACHINA_XLA_FFI_Api_Version{
        MACHINA_MACHINA_XLA_FFI_Api_Version_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        MACHINA_MACHINA_XLA_FFI_API_MAJOR,
        MACHINA_MACHINA_XLA_FFI_API_MINOR,
    },

    &internal_api,

    MACHINA_MACHINA_XLA_FFI_Error_Create,
    MACHINA_MACHINA_XLA_FFI_Error_GetMessage,
    MACHINA_MACHINA_XLA_FFI_Error_Destroy,
    MACHINA_MACHINA_XLA_FFI_Handler_Register,
    MACHINA_MACHINA_XLA_FFI_Stream_Get,
    MACHINA_MACHINA_XLA_FFI_TypeId_Register,
    MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get,
    MACHINA_MACHINA_XLA_FFI_State_Set,
    MACHINA_MACHINA_XLA_FFI_State_Get,
    MACHINA_MACHINA_XLA_FFI_DeviceMemory_Allocate,
    MACHINA_MACHINA_XLA_FFI_DeviceMemory_Free,
    MACHINA_MACHINA_XLA_FFI_ThreadPool_Schedule,
    MACHINA_MACHINA_XLA_FFI_ThreadPool_NumThreads,
    MACHINA_MACHINA_XLA_FFI_Future_Create,
    MACHINA_MACHINA_XLA_FFI_Future_SetAvailable,
    MACHINA_MACHINA_XLA_FFI_Future_SetError,
    MACHINA_MACHINA_XLA_FFI_RunId_Get,
    MACHINA_MACHINA_XLA_FFI_DeviceOrdinal_Get,
};

const MACHINA_MACHINA_XLA_FFI_Api* GetXlaFfiApi() { return &api; }

}  // namespace xla::ffi
