/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include <string>
#include <vector>

#include "machina/cc/framework/cc_op_gen.h"
#include "machina/cc/framework/cc_op_gen_util.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/framework/op_gen_lib.h"
#include "machina/core/lib/core/stringpiece.h"
#include "machina/core/lib/io/path.h"
#include "machina/core/lib/strings/str_util.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace cc_op {
namespace {

void PrintAllCCOps(const std::string& dot_h, const std::string& dot_cc,
                   bool include_internal,
                   const std::vector<string>& api_def_dirs) {
  OpList ops;
  absl::StatusOr<ApiDefMap> api_def_map =
      LoadOpsAndApiDefs(ops, include_internal, api_def_dirs);
  TF_CHECK_OK(api_def_map.status());
  api_def_map->UpdateDocs();
  WriteCCOps(ops, *api_def_map, dot_h, dot_cc);
}

}  // namespace
}  // namespace cc_op
}  // namespace machina

int main(int argc, char* argv[]) {
  machina::port::InitMain(argv[0], &argc, &argv);
  if (argc != 5) {
    for (int i = 1; i < argc; ++i) {
      fprintf(stderr, "Arg %d = %s\n", i, argv[i]);
    }
    fprintf(stderr,
            "Usage: %s out.h out.cc include_internal "
            "api_def_dirs1,api_def_dir2 ...\n"
            "  include_internal: 1 means include internal ops\n",
            argv[0]);
    exit(1);
  }

  bool include_internal = absl::string_view("1") == argv[3];
  std::vector<machina::string> api_def_dirs = machina::str_util::Split(
      argv[4], ",", machina::str_util::SkipEmpty());
  machina::cc_op::PrintAllCCOps(argv[1], argv[2], include_internal,
                                   api_def_dirs);
  return 0;
}
