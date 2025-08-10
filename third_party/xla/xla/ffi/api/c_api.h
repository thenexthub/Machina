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

#ifndef MACHINA_MACHINA_XLA_FFI_API_C_API_H_
#define MACHINA_MACHINA_XLA_FFI_API_C_API_H_

#include <stddef.h>
#include <stdint.h>

// XLA FFI C API follows PJRT API style for consistency. See `pjrt_c_api.h`.
// More details on versioning strategy and example version checks:
// https://github.com/machina/community/blob/master/rfcs/20200612-stream-executor-c-api/C_API_versioning_strategy.md

// Every struct passed across the C API boundary has its size as a member, and
// we use it as a sanity check for API compatibility.
#define MACHINA_MACHINA_XLA_FFI_STRUCT_SIZE(struct_type, last_field) \
  (offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field))

// Must update MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS with the new `last_field` after
// adding a new member to a struct.
#define MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(sname, last_field) \
  typedef struct sname sname;                           \
  enum { sname##_STRUCT_SIZE = MACHINA_MACHINA_XLA_FFI_STRUCT_SIZE(sname, last_field) }

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MACHINA_MACHINA_XLA_FFI_Api MACHINA_MACHINA_XLA_FFI_Api;                  // Forward declare
typedef struct MACHINA_MACHINA_XLA_FFI_InternalApi MACHINA_MACHINA_XLA_FFI_InternalApi;  // Forward declare

//===----------------------------------------------------------------------===//
// Extensions
//===----------------------------------------------------------------------===//

typedef enum {
  MACHINA_MACHINA_XLA_FFI_Extension_Metadata = 1,
} MACHINA_MACHINA_XLA_FFI_Extension_Type;

typedef struct MACHINA_MACHINA_XLA_FFI_Extension_Base {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Type type;
  struct MACHINA_MACHINA_XLA_FFI_Extension_Base* next;
} MACHINA_MACHINA_XLA_FFI_Extension_Base;

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Extension_Base, next);

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

// Incremented when an ABI-incompatible change is made to the interface.
//
// Major changes include:
// * Deleting a method or argument
// * Changing the type of an argument
// * Rearranging fields in the MACHINA_MACHINA_XLA_FFI_Api or argument structs
#define MACHINA_MACHINA_XLA_FFI_API_MAJOR 0

// Incremented when the interface is updated in a way that is potentially
// ABI-compatible with older versions, if supported by the caller and/or
// implementation.
//
// Callers can implement forwards compatibility by using MACHINA_MACHINA_XLA_FFI_Api_Version to
// check if the implementation is aware of newer interface additions.
//
// Implementations can implement backwards compatibility by using the
// `struct_size` fields to detect how many struct fields the caller is aware of.
//
// Minor changes include:
// * Adding a new field to the MACHINA_MACHINA_XLA_FFI_Api or argument structs
// * Renaming a method or argument (doesn't affect ABI)
#define MACHINA_MACHINA_XLA_FFI_API_MINOR 1

struct MACHINA_MACHINA_XLA_FFI_Api_Version {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;
  int major_version;  // out
  int minor_version;  // out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Api_Version, minor_version);

//===----------------------------------------------------------------------===//
// Error codes
//===----------------------------------------------------------------------===//

// XLA FFI error is a mechanism to communicate errors between XLA and XLA FFI
// via a set of C APIs. This is somewhat similar to type-erased version of
// absl::Status exposed via API with opaque pointers.
//
// Returning NULL error is equivalent to returning absl::OkStatus().
//
// Ownership of an MACHINA_MACHINA_XLA_FFI_Error is always transferred to the caller, and the
// caller is responsible for destroying it:
//
// (1) If the error is returned from an XLA FFI handler, the XLA runtime will
//     destroy it (XLA is the caller who calls into the handler implementation).
//
// (2) If the error is returned from an XLA FFI API call, the caller is
//     responsible for destroying it.
typedef struct MACHINA_MACHINA_XLA_FFI_Error MACHINA_MACHINA_XLA_FFI_Error;

// Codes are based on https://abseil.io/docs/cpp/guides/status-codes
typedef enum {
  MACHINA_MACHINA_XLA_FFI_Error_Code_OK = 0,
  MACHINA_MACHINA_XLA_FFI_Error_Code_CANCELLED = 1,
  MACHINA_MACHINA_XLA_FFI_Error_Code_UNKNOWN = 2,
  MACHINA_MACHINA_XLA_FFI_Error_Code_INVALID_ARGUMENT = 3,
  MACHINA_MACHINA_XLA_FFI_Error_Code_DEADLINE_EXCEEDED = 4,
  MACHINA_MACHINA_XLA_FFI_Error_Code_NOT_FOUND = 5,
  MACHINA_MACHINA_XLA_FFI_Error_Code_ALREADY_EXISTS = 6,
  MACHINA_MACHINA_XLA_FFI_Error_Code_PERMISSION_DENIED = 7,
  MACHINA_MACHINA_XLA_FFI_Error_Code_RESOURCE_EXHAUSTED = 8,
  MACHINA_MACHINA_XLA_FFI_Error_Code_FAILED_PRECONDITION = 9,
  MACHINA_MACHINA_XLA_FFI_Error_Code_ABORTED = 10,
  MACHINA_MACHINA_XLA_FFI_Error_Code_OUT_OF_RANGE = 11,
  MACHINA_MACHINA_XLA_FFI_Error_Code_UNIMPLEMENTED = 12,
  MACHINA_MACHINA_XLA_FFI_Error_Code_INTERNAL = 13,
  MACHINA_MACHINA_XLA_FFI_Error_Code_UNAVAILABLE = 14,
  MACHINA_MACHINA_XLA_FFI_Error_Code_DATA_LOSS = 15,
  MACHINA_MACHINA_XLA_FFI_Error_Code_UNAUTHENTICATED = 16
} MACHINA_MACHINA_XLA_FFI_Error_Code;

//===----------------------------------------------------------------------===//
// Error reporting APIs
//===----------------------------------------------------------------------===//

struct MACHINA_MACHINA_XLA_FFI_Error_Create_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;
  const char* message;
  MACHINA_MACHINA_XLA_FFI_Error_Code errc;
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Error_Create_Args, errc);

typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Error_Create(MACHINA_MACHINA_XLA_FFI_Error_Create_Args* args);

struct MACHINA_MACHINA_XLA_FFI_Error_GetMessage_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;
  MACHINA_MACHINA_XLA_FFI_Error* error;
  const char* message;  // out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Error_GetMessage_Args, message);

typedef void MACHINA_MACHINA_XLA_FFI_Error_GetMessage(MACHINA_MACHINA_XLA_FFI_Error_GetMessage_Args* args);

struct MACHINA_MACHINA_XLA_FFI_Error_Destroy_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;
  MACHINA_MACHINA_XLA_FFI_Error* error;
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Error_Destroy_Args, error);

typedef void MACHINA_MACHINA_XLA_FFI_Error_Destroy(MACHINA_MACHINA_XLA_FFI_Error_Destroy_Args* args);

//===----------------------------------------------------------------------===//
// DataType
//===----------------------------------------------------------------------===//

// This enum corresponds to xla::PrimitiveType enum defined in `xla_data.proto`.
// LINT.IfChange
typedef enum {
  MACHINA_MACHINA_XLA_FFI_DataType_INVALID = 0,
  MACHINA_MACHINA_XLA_FFI_DataType_PRED = 1,
  MACHINA_MACHINA_XLA_FFI_DataType_S1 = 30,
  MACHINA_MACHINA_XLA_FFI_DataType_S2 = 26,
  MACHINA_MACHINA_XLA_FFI_DataType_S4 = 21,
  MACHINA_MACHINA_XLA_FFI_DataType_S8 = 2,
  MACHINA_MACHINA_XLA_FFI_DataType_S16 = 3,
  MACHINA_MACHINA_XLA_FFI_DataType_S32 = 4,
  MACHINA_MACHINA_XLA_FFI_DataType_S64 = 5,
  MACHINA_MACHINA_XLA_FFI_DataType_U1 = 31,
  MACHINA_MACHINA_XLA_FFI_DataType_U2 = 27,
  MACHINA_MACHINA_XLA_FFI_DataType_U4 = 22,
  MACHINA_MACHINA_XLA_FFI_DataType_U8 = 6,
  MACHINA_MACHINA_XLA_FFI_DataType_U16 = 7,
  MACHINA_MACHINA_XLA_FFI_DataType_U32 = 8,
  MACHINA_MACHINA_XLA_FFI_DataType_U64 = 9,
  MACHINA_MACHINA_XLA_FFI_DataType_F16 = 10,
  MACHINA_MACHINA_XLA_FFI_DataType_F32 = 11,
  MACHINA_MACHINA_XLA_FFI_DataType_F64 = 12,
  MACHINA_MACHINA_XLA_FFI_DataType_BF16 = 16,
  MACHINA_MACHINA_XLA_FFI_DataType_C64 = 15,
  MACHINA_MACHINA_XLA_FFI_DataType_C128 = 18,
  MACHINA_MACHINA_XLA_FFI_DataType_TOKEN = 17,
  MACHINA_MACHINA_XLA_FFI_DataType_F8E5M2 = 19,
  MACHINA_MACHINA_XLA_FFI_DataType_F8E3M4 = 29,
  MACHINA_MACHINA_XLA_FFI_DataType_F8E4M3 = 28,
  MACHINA_MACHINA_XLA_FFI_DataType_F8E4M3FN = 20,
  MACHINA_MACHINA_XLA_FFI_DataType_F8E4M3B11FNUZ = 23,
  MACHINA_MACHINA_XLA_FFI_DataType_F8E5M2FNUZ = 24,
  MACHINA_MACHINA_XLA_FFI_DataType_F8E4M3FNUZ = 25,
  MACHINA_MACHINA_XLA_FFI_DataType_F4E2M1FN = 32,
  MACHINA_MACHINA_XLA_FFI_DataType_F8E8M0FNU = 33,
} MACHINA_MACHINA_XLA_FFI_DataType;
// LINT.ThenChange(ffi_test.cc)

//===----------------------------------------------------------------------===//
// Builtin argument types
//===----------------------------------------------------------------------===//

struct MACHINA_MACHINA_XLA_FFI_Buffer {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_DataType dtype;
  void* data;
  int64_t rank;
  int64_t* dims;  // length == rank
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Buffer, dims);

typedef enum {
  MACHINA_MACHINA_XLA_FFI_ArgType_BUFFER = 1,
} MACHINA_MACHINA_XLA_FFI_ArgType;

//===----------------------------------------------------------------------===//
// Builtin result types
//===----------------------------------------------------------------------===//

typedef enum {
  MACHINA_MACHINA_XLA_FFI_RetType_BUFFER = 1,
} MACHINA_MACHINA_XLA_FFI_RetType;

//===----------------------------------------------------------------------===//
// Builtin attribute types
//===----------------------------------------------------------------------===//

typedef enum {
  MACHINA_MACHINA_XLA_FFI_AttrType_ARRAY = 1,
  MACHINA_MACHINA_XLA_FFI_AttrType_DICTIONARY = 2,
  MACHINA_MACHINA_XLA_FFI_AttrType_SCALAR = 3,
  MACHINA_MACHINA_XLA_FFI_AttrType_STRING = 4,
} MACHINA_MACHINA_XLA_FFI_AttrType;

//===----------------------------------------------------------------------===//
// Execution context
//===----------------------------------------------------------------------===//

// Execution context provides access to per-invocation state.
typedef struct MACHINA_MACHINA_XLA_FFI_ExecutionContext MACHINA_MACHINA_XLA_FFI_ExecutionContext;

//===----------------------------------------------------------------------===//
// Primitives
//===----------------------------------------------------------------------===//

// TypeId uniquely identifies a user-defined type in a given XLA FFI instance.
typedef struct MACHINA_MACHINA_XLA_FFI_TypeId {
  int64_t type_id;
} MACHINA_MACHINA_XLA_FFI_TypeId;

// We use byte spans to pass strings to handlers because strings might not be
// null terminated, and even if they are, looking for a null terminator can
// become very expensive in tight loops.
typedef struct MACHINA_MACHINA_XLA_FFI_ByteSpan {
  const char* ptr;
  size_t len;
} MACHINA_MACHINA_XLA_FFI_ByteSpan;

// A struct to pass a scalar value to FFI handler.
typedef struct MACHINA_MACHINA_XLA_FFI_Scalar {
  MACHINA_MACHINA_XLA_FFI_DataType dtype;
  void* value;
} MACHINA_MACHINA_XLA_FFI_Scalar;

// A struct to pass a dense array to FFI handler.
typedef struct MACHINA_MACHINA_XLA_FFI_Array {
  MACHINA_MACHINA_XLA_FFI_DataType dtype;
  size_t size;
  void* data;
} MACHINA_MACHINA_XLA_FFI_Array;

//===----------------------------------------------------------------------===//
// Future
//===----------------------------------------------------------------------===//

// XLA FFI future is a mechanism to signal a result of asynchronous computation
// (FFI handler) to the XLA runtime. It is similar to `std::future<void>` in C++
// standard library, and implemented on top of `tsl::AsyncValue` in XLA runtime.
//
// XLA FFI users should use `Future` and `Promise` types defined in `xla::ffi`
// namespace (see `ffi/api/ffi.h`), instead of using this API directly.
typedef struct MACHINA_MACHINA_XLA_FFI_Future MACHINA_MACHINA_XLA_FFI_Future;

struct MACHINA_MACHINA_XLA_FFI_Future_Create_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;
  MACHINA_MACHINA_XLA_FFI_Future* future;  // out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Future_Create_Args, extension_start);

typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Future_Create(MACHINA_MACHINA_XLA_FFI_Future_Create_Args* args);

struct MACHINA_MACHINA_XLA_FFI_Future_SetAvailable_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;
  MACHINA_MACHINA_XLA_FFI_Future* future;
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Future_SetAvailable_Args, future);

typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Future_SetAvailable(
    MACHINA_MACHINA_XLA_FFI_Future_SetAvailable_Args* args);

struct MACHINA_MACHINA_XLA_FFI_Future_SetError_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;
  MACHINA_MACHINA_XLA_FFI_Future* future;
  MACHINA_MACHINA_XLA_FFI_Error* error;  // ownership is transferred to the XLA runtime
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Future_SetError_Args, error);

typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Future_SetError(
    MACHINA_MACHINA_XLA_FFI_Future_SetError_Args* args);

//===----------------------------------------------------------------------===//
// Call frame
//===----------------------------------------------------------------------===//

// XLA runtime has multiple execution stages and it is possible to run
// different handlers for each stage:
//
// (1) Instantiate - called when FFI handler is instantiated as a part of XLA
//     executable instantiation. Every call site will have its own "instance" of
//     the FFI handler, and it is possible to attach an arbitrary user-defined
//     state to the FFI handler instance, and get it back in other execution
//     stages. Constructed state owned by the XLA runtime and destructed
//     together with a parent executable.
//
// (2) Prepare - called before the execution to let FFI handlers to prepare
//     for the execution and request resources from runtime, i.e. in XLA:GPU
//     we use prepare stage to request collective cliques.
//
// (3) Initialize - called before the execution after acquiring all the
//     resources requested in the prepare stage.
//
// (4) Execute - called when FFI handler is executed. Note that FFI handler
//     can be called as a part of command buffer capture (CUDA graph capture
//     on GPU backend) and argument buffers might contain uninitialized
//     values in this case.
//
// XLA program (HLO module) compiled to an XLA executable that can be executed
// on any device accessible to the process, and by extension FFI handlers are
// not instantiated for any particular device, but for a process. FFI handlers
// running at instantiation stage do not have access to the underlying device
// (memory allocation, stream, etc.) and arguments, however they can access
// execution context and attributes.
//
// It is undefined behavior to access argument buffers in prepare and initialize
// stages as they might not be initialized yet. However it is safe to use memory
// address as it is assigned ahead of time by buffer assignment.
typedef enum {
  MACHINA_MACHINA_XLA_FFI_ExecutionStage_INSTANTIATE = 0,
  MACHINA_MACHINA_XLA_FFI_ExecutionStage_PREPARE = 1,
  MACHINA_MACHINA_XLA_FFI_ExecutionStage_INITIALIZE = 2,
  MACHINA_MACHINA_XLA_FFI_ExecutionStage_EXECUTE = 3,
} MACHINA_MACHINA_XLA_FFI_ExecutionStage;

struct MACHINA_MACHINA_XLA_FFI_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  int64_t size;
  MACHINA_MACHINA_XLA_FFI_ArgType* types;  // length == size
  void** args;             // length == size
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Args, args);

struct MACHINA_MACHINA_XLA_FFI_Rets {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  int64_t size;
  MACHINA_MACHINA_XLA_FFI_RetType* types;  // length == size
  void** rets;             // length == size
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Rets, rets);

// FFI handler attributes are always sorted by name, so that the handler can
// rely on binary search to look up attributes by name.
struct MACHINA_MACHINA_XLA_FFI_Attrs {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  int64_t size;
  MACHINA_MACHINA_XLA_FFI_AttrType* types;   // length == size
  MACHINA_MACHINA_XLA_FFI_ByteSpan** names;  // length == size
  void** attrs;              // length == size
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Attrs, attrs);

struct MACHINA_MACHINA_XLA_FFI_CallFrame {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  const MACHINA_MACHINA_XLA_FFI_Api* api;
  MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx;
  MACHINA_MACHINA_XLA_FFI_ExecutionStage stage;
  MACHINA_MACHINA_XLA_FFI_Args args;
  MACHINA_MACHINA_XLA_FFI_Rets rets;
  MACHINA_MACHINA_XLA_FFI_Attrs attrs;

  // XLA FFI handler implementation can use `future` to signal a result of
  // asynchronous computation to the XLA runtime. XLA runtime will keep all
  // arguments, results and attributes alive until `future` is completed.
  MACHINA_MACHINA_XLA_FFI_Future* future;  // out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_CallFrame, attrs);

//===----------------------------------------------------------------------===//
// FFI handler
//===----------------------------------------------------------------------===//

// External functions registered with XLA as FFI handlers.
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Handler(MACHINA_MACHINA_XLA_FFI_CallFrame* call_frame);

// XLA FFI handlers for execution stages (see MACHINA_MACHINA_XLA_FFI_ExecutionStage).
typedef struct MACHINA_MACHINA_XLA_FFI_Handler_Bundle {
  MACHINA_MACHINA_XLA_FFI_Handler* instantiate;  // optional
  MACHINA_MACHINA_XLA_FFI_Handler* prepare;      // optional
  MACHINA_MACHINA_XLA_FFI_Handler* initialize;   // optional
  MACHINA_MACHINA_XLA_FFI_Handler* execute;      // required
} MACHINA_MACHINA_XLA_FFI_Handler_Bundle;

enum MACHINA_MACHINA_XLA_FFI_Handler_TraitsBits {
  // Calls to FFI handler are safe to trace into the command buffer. It means
  // that calls to FFI handler always launch exactly the same device operations
  // (can depend on attribute values) that can be captured and then replayed.
  MACHINA_MACHINA_XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE = 1u << 0,
};

typedef uint32_t MACHINA_MACHINA_XLA_FFI_Handler_Traits;

struct MACHINA_MACHINA_XLA_FFI_Handler_Register_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ByteSpan name;
  MACHINA_MACHINA_XLA_FFI_ByteSpan platform;
  MACHINA_MACHINA_XLA_FFI_Handler_Bundle bundle;
  MACHINA_MACHINA_XLA_FFI_Handler_Traits traits;
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Handler_Register_Args, traits);

typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Handler_Register(
    MACHINA_MACHINA_XLA_FFI_Handler_Register_Args* args);

//===----------------------------------------------------------------------===//
// TypeId
//===----------------------------------------------------------------------===//

#define MACHINA_MACHINA_XLA_FFI_UNKNOWN_TYPE_ID MACHINA_MACHINA_XLA_FFI_TypeId{0}

struct MACHINA_MACHINA_XLA_FFI_TypeId_Register_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ByteSpan name;
  MACHINA_MACHINA_XLA_FFI_TypeId* type_id;  // in-out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_TypeId_Register_Args, type_id);

// Registers user type `name` with XLA. If type id is `MACHINA_MACHINA_XLA_FFI_UNKNOWN_TYPE_ID`,
// XLA will assign a unique type id and return it in `type_id` out argument,
// otherwise XLA will verify that type id is unique and matches the type id of
// the type registered with the same `name` earlier.
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_TypeId_Register(
    MACHINA_MACHINA_XLA_FFI_TypeId_Register_Args* args);

//===----------------------------------------------------------------------===//
// ExecutionContext
//===----------------------------------------------------------------------===//

struct MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx;
  MACHINA_MACHINA_XLA_FFI_TypeId* type_id;
  void* data;  // out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get_Args, data);

// Returns an opaque data from the execution context for a given type id.
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get(
    MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get_Args* args);

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

struct MACHINA_MACHINA_XLA_FFI_State_Set_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx;
  MACHINA_MACHINA_XLA_FFI_TypeId* type_id;
  void* state;
  void (*deleter)(void* state);
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_State_Set_Args, deleter);

// Sets execution state to the `state` of type `type_id`. Returns an error if
// state already set.
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_State_Set(MACHINA_MACHINA_XLA_FFI_State_Set_Args* args);

struct MACHINA_MACHINA_XLA_FFI_State_Get_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx;
  MACHINA_MACHINA_XLA_FFI_TypeId* type_id;
  void* state;  // out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_State_Get_Args, state);

// Gets execution state of type `type_id`. Returns an error if state is not set,
// or set with a state of a different type.
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_State_Get(MACHINA_MACHINA_XLA_FFI_State_Get_Args* args);

//===----------------------------------------------------------------------===//
// Stream
//===----------------------------------------------------------------------===//

struct MACHINA_MACHINA_XLA_FFI_Stream_Get_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx;
  void* stream;  // out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Stream_Get_Args, stream);

// Returns an underling platform-specific stream via out argument, i.e. for CUDA
// platform it returns `CUstream` (same as `cudaStream`).
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_Stream_Get(MACHINA_MACHINA_XLA_FFI_Stream_Get_Args* args);

//===----------------------------------------------------------------------===//
// Device memory allocation
//===----------------------------------------------------------------------===//

struct MACHINA_MACHINA_XLA_FFI_DeviceMemory_Allocate_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx;
  size_t size;
  size_t alignment;
  void* data;  // out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_DeviceMemory_Allocate_Args, data);

// Allocates a block of memory on the device bound to the execution context.
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_DeviceMemory_Allocate(
    MACHINA_MACHINA_XLA_FFI_DeviceMemory_Allocate_Args* args);

struct MACHINA_MACHINA_XLA_FFI_DeviceMemory_Free_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx;
  size_t size;
  void* data;
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_DeviceMemory_Free_Args, data);

// Frees previously allocated device memory.
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_DeviceMemory_Free(
    MACHINA_MACHINA_XLA_FFI_DeviceMemory_Free_Args* args);

//===----------------------------------------------------------------------===//
// ThreadPool
//===----------------------------------------------------------------------===//

// A function pointer for a task to be scheduled on a thread pool. XLA runtime
// will call this function with a user-defined `data` pointer on one of the
// runtime-managed threads. For XLA:CPU backends the task will be invoked on
// a thread pool that runs all compute tasks (Eigen thread pool).
//
// IMPORTANT: Users must not rely on any particular execution order or the
// number of available threads. Tasks can be executed in the caller thread, or
// in a thread pool with size `1`, and it is unsafe to assume that all scheduled
// tasks can be executed in parallel.
typedef void MACHINA_MACHINA_XLA_FFI_Task(void* data);

struct MACHINA_MACHINA_XLA_FFI_ThreadPool_Schedule_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx;
  MACHINA_MACHINA_XLA_FFI_Task* task;
  void* data;
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_ThreadPool_Schedule_Args, data);

// Schedules a task to be executed on a thread pool managed by XLA runtime.
// Returns an error if thread pool is not available.
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_ThreadPool_Schedule(
    MACHINA_MACHINA_XLA_FFI_ThreadPool_Schedule_Args* args);

struct MACHINA_MACHINA_XLA_FFI_ThreadPool_NumThreads_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx;
  int64_t* num_threads;  // out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_ThreadPool_NumThreads_Args, num_threads);

// Returns the number of threads in the thread pool managed by XLA runtime.
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_ThreadPool_NumThreads(
    MACHINA_MACHINA_XLA_FFI_ThreadPool_NumThreads_Args* args);

//===----------------------------------------------------------------------===//
// RunId
//===----------------------------------------------------------------------===//

// RunId is a unique identifier for a particular "logical execution" of an XLA
// model.
//
// A logical execution might encompass multiple executions of one or more
// HloModules. Runs that are part of the same logical execution can communicate
// via collective ops, whereas runs that are part of different logical
// executions are isolated.
//
// Corresponds to `::xla::RunId` (see `xla/executable_run_options.h`).

struct MACHINA_MACHINA_XLA_FFI_RunId_Get_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx;
  int64_t run_id;  // out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_RunId_Get_Args, run_id);

// Returns a unique identifier for the current logical execution.
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_RunId_Get(MACHINA_MACHINA_XLA_FFI_RunId_Get_Args* args);

//===----------------------------------------------------------------------===//
// DeviceOrdinal
//===----------------------------------------------------------------------===//

struct MACHINA_MACHINA_XLA_FFI_DeviceOrdinal_Get_Args {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_ExecutionContext* ctx;
  int32_t device_ordinal;  // out
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_DeviceOrdinal_Get_Args, device_ordinal);

// Returns a unique identifier for the current logical execution.
typedef MACHINA_MACHINA_XLA_FFI_Error* MACHINA_MACHINA_XLA_FFI_DeviceOrdinal_Get(
    MACHINA_MACHINA_XLA_FFI_DeviceOrdinal_Get_Args* args);

//===----------------------------------------------------------------------===//
// Metadata extension
//===----------------------------------------------------------------------===//

struct MACHINA_MACHINA_XLA_FFI_Metadata {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Api_Version api_version;
  MACHINA_MACHINA_XLA_FFI_Handler_Traits traits;
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Metadata, traits);

struct MACHINA_MACHINA_XLA_FFI_Metadata_Extension {
  MACHINA_MACHINA_XLA_FFI_Extension_Base extension_base;
  MACHINA_MACHINA_XLA_FFI_Metadata* metadata;
};

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Metadata_Extension, metadata);

//===----------------------------------------------------------------------===//
// API access
//===----------------------------------------------------------------------===//

#define _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(fn_type) fn_type* fn_type

struct MACHINA_MACHINA_XLA_FFI_Api {
  size_t struct_size;
  MACHINA_MACHINA_XLA_FFI_Extension_Base* extension_start;

  MACHINA_MACHINA_XLA_FFI_Api_Version api_version;
  MACHINA_MACHINA_XLA_FFI_InternalApi* internal_api;

  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_Error_Create);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_Error_GetMessage);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_Error_Destroy);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_Handler_Register);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_Stream_Get);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_TypeId_Register);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_ExecutionContext_Get);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_State_Set);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_State_Get);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_DeviceMemory_Allocate);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_DeviceMemory_Free);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_ThreadPool_Schedule);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_ThreadPool_NumThreads);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_Future_Create);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_Future_SetAvailable);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_Future_SetError);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_RunId_Get);
  _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD(MACHINA_MACHINA_XLA_FFI_DeviceOrdinal_Get);
};

#undef _MACHINA_MACHINA_XLA_FFI_API_STRUCT_FIELD

MACHINA_MACHINA_XLA_FFI_DEFINE_STRUCT_TRAITS(MACHINA_MACHINA_XLA_FFI_Api, MACHINA_MACHINA_XLA_FFI_DeviceOrdinal_Get);

const MACHINA_MACHINA_XLA_FFI_Api* MACHINA_MACHINA_XLA_FFI_GetApi();

#ifdef __cplusplus
}
#endif

#endif  // MACHINA_MACHINA_XLA_FFI_API_C_API_H_
