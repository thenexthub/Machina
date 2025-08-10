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

#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "machina/compiler/aot/compile.h"
#include "machina/compiler/aot/flags.h"
#include "machina/compiler/tf2xla/tf2xla.pb.h"
#include "machina/xla/debug_options_flags.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/platform/logging.h"
#include "machina/core/util/command_line_flags.h"

namespace machina {
namespace tfcompile {

const char kUsageHeader[] =
    "tfcompile performs ahead-of-time compilation of a TensorFlow graph,\n"
    "resulting in an object file compiled for your target architecture, and a\n"
    "header file that gives access to the functionality in the object file.\n"
    "A typical invocation looks like this:\n"
    "\n"
    "   $ tfcompile --graph=mygraph.pb --config=myfile.pbtxt "
    "--cpp_class=\"mynamespace::MyComputation\"\n"
    "\n";

}  // end namespace tfcompile
}  // end namespace machina

int main(int argc, char** argv) {
  machina::tfcompile::MainFlags flags;
#ifndef __s390x__
  flags.target_triple = "x86_64-pc-linux";
#endif
  flags.out_function_object = "out_model.o";
  flags.out_metadata_object = "out_helper.o";
  flags.out_header = "out.h";
  flags.entry_point = "entry";
  flags.debug_info_path_begin_marker = "";

  // Note that tfcompile.bzl's tf_library macro sets fast math flags as that is
  // generally the preferred case.
  std::vector<machina::Flag> flag_list;
  AppendMainFlags(&flag_list, &flags);
  xla::AppendDebugOptionsFlags(&flag_list);

  machina::string usage = machina::tfcompile::kUsageHeader;
  usage += machina::Flags::Usage(argv[0], flag_list);
  if (argc > 1 && absl::string_view(argv[1]) == "--help") {
    std::cerr << usage << "\n";
    return 0;
  }
  bool parsed_flags_ok = machina::Flags::Parse(&argc, argv, flag_list);
  QCHECK(parsed_flags_ok) << "\n" << usage;

  machina::port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(argc == 1) << "\nERROR: This command does not take any arguments "
                       "other than flags. See --help.\n\n";
  absl::Status status = machina::tfcompile::Main(flags);
  if (status.code() == absl::StatusCode::kInvalidArgument) {
    std::cerr << "INVALID ARGUMENTS: " << status.message() << "\n\n";
    return 1;
  } else {
    TF_QCHECK_OK(status);
  }
  return 0;
}
