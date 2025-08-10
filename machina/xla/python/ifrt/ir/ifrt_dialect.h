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

#ifndef MACHINA_XLAPYTHON_IFRT_IR_IFRT_DIALECT_H_
#define MACHINA_XLAPYTHON_IFRT_IR_IFRT_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "machina/xla/pjrt/layout_mode.h"
#include "machina/xla/python/ifrt/ir/ifrt_interfaces.h"
#include "machina/xla/python/ifrt/ir/sharding_param.h"
#include "machina/xla/python/ifrt/memory.h"

// Generated definitions.
#include "machina/xla/python/ifrt/ir/ifrt_dialect.h.inc"  // IWYU pragma: export
#define GET_ATTRDEF_CLASSES
#include "machina/xla/python/ifrt/ir/ifrt_attrs.h.inc"  // IWYU pragma: export
#define GET_TYPEDEF_CLASSES
#include "machina/xla/python/ifrt/ir/ifrt_types.h.inc"  // IWYU pragma: export

#endif  // MACHINA_XLAPYTHON_IFRT_IR_IFRT_DIALECT_H_
