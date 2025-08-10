// Main executable to generate op fuzzers

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
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/cc/framework/cc_op_gen_util.h"
#include "machina/cc/framework/fuzzing/cc_op_fuzz_gen.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/core/framework/api_def.pb.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/framework/op_gen_lib.h"
#include "machina/core/lib/core/stringpiece.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/file_system.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/platform/str_util.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace cc_op {
namespace {

void WriteAllFuzzers(string root_location, std::vector<string> api_def_dirs,
                     std::vector<string> op_names) {
  OpList ops;
  absl::StatusOr<ApiDefMap> api_def_map =
      LoadOpsAndApiDefs(ops, false, api_def_dirs);

  TF_CHECK_OK(api_def_map.status());

  Env* env = Env::Default();
  absl::Status status;
  std::unique_ptr<WritableFile> fuzz_file = nullptr;
  for (const OpDef& op_def : ops.op()) {
    if (std::find(op_names.begin(), op_names.end(), op_def.name()) ==
        op_names.end())
      continue;

    const ApiDef* api_def = api_def_map->GetApiDef(op_def.name());
    if (api_def == nullptr) {
      continue;
    }

    OpInfo op_info(op_def, *api_def, std::vector<string>());
    status.Update(env->NewWritableFile(
        root_location + "/" + op_def.name() + "_fuzz.cc", &fuzz_file));
    status.Update(
        fuzz_file->Append(WriteSingleFuzzer(op_info, OpFuzzingIsOk(op_info))));
    status.Update(fuzz_file->Close());
  }
  TF_CHECK_OK(status);
}

}  // namespace
}  // namespace cc_op
}  // namespace machina

int main(int argc, char* argv[]) {
  machina::port::InitMain(argv[0], &argc, &argv);
  if (argc != 4) {
    for (int i = 1; i < argc; ++i) {
      fprintf(stderr, "Arg %d = %s\n", i, argv[i]);
    }
    fprintf(stderr, "Usage: %s location api_def1,api_def2 op1,op2,op3\n",
            argv[0]);
    exit(1);
  }
  for (int i = 1; i < argc; ++i) {
    fprintf(stdout, "Arg %d = %s\n", i, argv[i]);
  }
  std::vector<machina::string> api_def_srcs = machina::str_util::Split(
      argv[2], ",", machina::str_util::SkipEmpty());
  std::vector<machina::string> op_names = machina::str_util::Split(
      argv[3], ",", machina::str_util::SkipEmpty());
  machina::cc_op::WriteAllFuzzers(argv[1], api_def_srcs, op_names);
  return 0;
}
