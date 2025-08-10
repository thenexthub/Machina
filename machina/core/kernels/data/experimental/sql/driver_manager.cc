/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include "machina/core/kernels/data/experimental/sql/driver_manager.h"

#include <memory>

#include "machina/core/kernels/data/experimental/sql/sqlite_query_connection.h"

namespace machina {
namespace data {
namespace experimental {
namespace sql {

std::unique_ptr<QueryConnection> DriverManager::CreateQueryConnection(
    const string& driver_name) {
  if (driver_name == "sqlite") {
    return std::make_unique<SqliteQueryConnection>();
  } else {  // TODO(b/64276826, b/64276995) Add support for other db types.
            // Change to registry pattern.
    return nullptr;
  }
}

}  // namespace sql
}  // namespace experimental
}  // namespace data
}  // namespace machina
