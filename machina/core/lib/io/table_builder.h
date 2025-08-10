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

// TableBuilder provides the interface used to build a Table
// (an immutable and sorted map from keys to values).
//
// Multiple threads can invoke const methods on a TableBuilder without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same TableBuilder must use
// external synchronization.

#ifndef MACHINA_CORE_LIB_IO_TABLE_BUILDER_H_
#define MACHINA_CORE_LIB_IO_TABLE_BUILDER_H_

#include "machina/xla/tsl/lib/io/table_builder.h"
#include "machina/core/lib/io/table_options.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/stringpiece.h"

namespace machina {
namespace table {
using tsl::table::TableBuilder;  // NOLINT(misc-unused-using-decls)
}
}  // namespace machina

#endif  // MACHINA_CORE_LIB_IO_TABLE_BUILDER_H_
