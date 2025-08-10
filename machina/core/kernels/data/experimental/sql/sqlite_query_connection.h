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
#ifndef MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_SQL_SQLITE_QUERY_CONNECTION_H_
#define MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_SQL_SQLITE_QUERY_CONNECTION_H_

#include <memory>

#include "machina/core/kernels/data/experimental/sql/query_connection.h"
#include "machina/core/lib/db/sqlite.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace data {
namespace experimental {
namespace sql {

class SqliteQueryConnection : public QueryConnection {
 public:
  SqliteQueryConnection();
  ~SqliteQueryConnection() override;
  absl::Status Open(const string& data_source_name, const string& query,
                    const DataTypeVector& output_types) override;
  absl::Status Close() override;
  absl::Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                       bool* end_of_sequence) override;

 private:
  // Prepares the query string `query_`.
  absl::Status PrepareQuery();
  // Fills `tensor` with the column_index_th element of the current row of
  // `stmt_`.
  void FillTensorWithResultSetEntry(const DataType& data_type, int column_index,
                                    Tensor* tensor);
  Sqlite* db_ = nullptr;
  SqliteStatement stmt_;
  int column_count_ = 0;
  string query_;
  DataTypeVector output_types_;
};

}  // namespace sql
}  // namespace experimental
}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_SQL_SQLITE_QUERY_CONNECTION_H_
