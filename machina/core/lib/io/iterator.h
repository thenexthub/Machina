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

// An iterator yields a sequence of key/value pairs from a source.
// The following class defines the interface.  Multiple implementations
// are provided by this library.  In particular, iterators are provided
// to access the contents of a Table or a DB.
//
// Multiple threads can invoke const methods on an Iterator without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same Iterator must use
// external synchronization.

#ifndef MACHINA_CORE_LIB_IO_ITERATOR_H_
#define MACHINA_CORE_LIB_IO_ITERATOR_H_

#include "machina/xla/tsl/lib/io/iterator.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/stringpiece.h"

namespace machina {
namespace table {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::table::Iterator;
using tsl::table::NewEmptyIterator;
using tsl::table::NewErrorIterator;
// NOLINTEND(misc-unused-using-decls)
}  // namespace table
}  // namespace machina

#endif  // MACHINA_CORE_LIB_IO_ITERATOR_H_
