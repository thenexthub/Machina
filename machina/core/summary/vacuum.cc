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
#include <iostream>

#include "absl/log/log.h"
#include "machina/core/lib/db/sqlite.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/util/command_line_flags.h"

namespace machina {
namespace {

void Vacuum(const char* path) {
  LOG(INFO) << "Opening SQLite DB: " << path;
  Sqlite* db;
  TF_CHECK_OK(Sqlite::Open(path, SQLITE_OPEN_READWRITE, &db));
  core::ScopedUnref db_unref(db);

  // TODO(jart): Maybe defragment rowids on Tensors.
  // TODO(jart): Maybe LIMIT deletes and incremental VACUUM.

  // clang-format off

  LOG(INFO) << "Deleting orphaned Experiments";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Experiments
    WHERE
      user_id IS NOT NULL
      AND user_id NOT IN (SELECT user_id FROM Users)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned Runs";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Runs
    WHERE
      experiment_id IS NOT NULL
      AND experiment_id NOT IN (SELECT experiment_id FROM Experiments)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned Tags";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Tags
    WHERE
      run_id IS NOT NULL
      AND run_id NOT IN (SELECT run_id FROM Runs)
  )sql").StepAndResetOrDie();

  // TODO(jart): What should we do if plugins define non-tag tensor series?
  LOG(INFO) << "Deleting orphaned Tensors";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Tensors
    WHERE
      series IS NOT NULL
      AND series NOT IN (SELECT tag_id FROM Tags)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned TensorStrings";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      TensorStrings
    WHERE
      tensor_rowid NOT IN (SELECT rowid FROM Tensors)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned Graphs";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Graphs
    WHERE
      run_id IS NOT NULL
      AND run_id NOT IN (SELECT run_id FROM Runs)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned Nodes";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Nodes
    WHERE
      graph_id NOT IN (SELECT graph_id FROM Graphs)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned NodeInputs";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      NodeInputs
    WHERE
      graph_id NOT IN (SELECT graph_id FROM Graphs)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Running VACUUM";
  db->PrepareOrDie("VACUUM").StepAndResetOrDie();

  // clang-format on
}

int main(int argc, char* argv[]) {
  string usage = Flags::Usage(argv[0], {});
  bool parse_result = Flags::Parse(&argc, argv, {});
  if (!parse_result) {
    std::cerr << "The vacuum tool rebuilds SQLite database files created by\n"
              << "SummaryDbWriter, which makes them smaller.\n\n"
              << "This means deleting orphaned rows and rebuilding b-tree\n"
              << "pages so empty space from deleted rows is cleared. Any\n"
              << "superfluous padding of Tensor BLOBs is also removed.\n\n"
              << usage;
    return -1;
  }
  port::InitMain(argv[0], &argc, &argv);
  if (argc < 2 || argv[1][0] == '-') {
    std::cerr << "Need at least one SQLite DB path.\n";
    return -1;
  }
  for (int i = 1; i < argc; ++i) {
    Vacuum(argv[i]);
  }
  return 0;
}

}  // namespace
}  // namespace machina

int main(int argc, char* argv[]) { return machina::main(argc, argv); }
