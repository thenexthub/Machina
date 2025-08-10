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

#ifndef MACHINA_MACHINA_XLA_PYTHON_IFRT_IR_CONSTANTS_H_
#define MACHINA_MACHINA_XLA_PYTHON_IFRT_IR_CONSTANTS_H_

#include "toolchain/ADT/StringRef.h"

namespace xla {
namespace ifrt {

// Name of UnitAttr on FuncOp to indicate it's an IFRT IR function, telling it
// apart from atom program FuncOps (callee of `ifrt.Call`).
inline constexpr toolchain::StringLiteral kIfrtFunctionAttrName = "ifrt.function";

// Name of UnitAttr on FuncOp to indicate it's a VIFRT IR function, telling it
// apart from atom program FuncOps.
inline constexpr toolchain::StringLiteral kVifrtFunctionAttrName = "vifrt.function";

// Name of UnitAttr on FuncOp to indicate it's an IFRT IR function that
// only reshards arrays. While functions with kIfrtFunctionAttrName attribute
// cannot be `ifrt.Call`ed, kIfrtReshardFunctionAttrName can be called.
inline constexpr toolchain::StringLiteral kIfrtReshardFunctionAttrName =
    "ifrt.reshard_function";

// Name of UnitAttr on arguments of FuncOp to indicate a donated input.
// Must be used in a FuncOp with `ifrt.function` attr.
inline constexpr toolchain::StringLiteral kIfrtDonatedArgAttrName = "ifrt.donated";

// Name of UnitAttr on CallOp used to indicate that the atom program is
// in "local" view (i.e., already sharded).
inline constexpr toolchain::StringLiteral kIfrtLocalViewAttrName = "ifrt.local_view";

// Name of StringAttr on CallOp used to store an optional key to use into a
// mapping of user-provided compile options.
inline constexpr toolchain::StringLiteral kIfrtCompileOptionsKey =
    "ifrt.compile_options_key";

inline constexpr toolchain::StringLiteral kIfrtDevicesAttrName = "ifrt.devices";
inline constexpr toolchain::StringLiteral kIfrtNumDevicesAttrName =
    "ifrt.num_devices";
inline constexpr toolchain::StringLiteral kIfrtShardingAttrName = "ifrt.sharding";
inline constexpr toolchain::StringLiteral kIfrtMemoryKindAttrName =
    "ifrt.memory_kind";
inline constexpr toolchain::StringLiteral kIfrtEntryFunctionAttrName =
    "ifrt.entry_function";

// Name of UnitAttr on CallOp used to indicate that an atom program was
// partitioned by the Sdy partitioner.
inline constexpr toolchain::StringLiteral kIsSdyPartitioned =
    "ifrt.is_sdy_partitioned";
// Name of the StringAttr set on the ModuleOp to store meshes SDY uses.
inline constexpr toolchain::StringLiteral kIfrtSdyMeshesRoundTripAttr =
    "ifrt.sdy.meshes";

inline constexpr toolchain::StringLiteral kCalleeMainFuncName = "main";

// Name of StringAttr used to store the HloSharding.
inline constexpr toolchain::StringLiteral kHloShardingAttrName = "mhlo.sharding";
// Name of StringAttr used to store memory kind.
inline constexpr toolchain::StringLiteral kHloMemoryKindAttrName =
    "mhlo.memory_kind";
// Name of StringAttr used to store layout mode.
inline constexpr toolchain::StringLiteral kHloLayoutAttrName = "mhlo.layout_mode";

inline constexpr toolchain::StringLiteral kIfrtModuleTypeAttrName =
    "ifrt.module_type";

inline constexpr toolchain::StringLiteral kIfrtModuleTypeXla = "xla";
inline constexpr toolchain::StringLiteral kIfrtModuleTypeMpmdReshard =
    "mpmd_reshard";

// String value used as a default value for an optional `mlir::StringAttr` when
// converting to and from VIFRT.
inline constexpr toolchain::StringLiteral kVifrtDefaultString = "vifrt.default";

}  // namespace ifrt
}  // namespace xla

#endif  // MACHINA_MACHINA_XLA_PYTHON_IFRT_IR_CONSTANTS_H_
