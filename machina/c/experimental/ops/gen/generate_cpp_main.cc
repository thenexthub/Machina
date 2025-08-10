/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "machina/c/experimental/ops/gen/common/path_config.h"
#include "machina/c/experimental/ops/gen/cpp/cpp_generator.h"
#include "machina/c/experimental/ops/gen/cpp/renderers/cpp_config.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/platform/logging.h"
#include "machina/core/util/command_line_flags.h"

using machina::string;

namespace generator = machina::generator;

namespace {
class MainConfig {
 public:
  void InitMain(int* argc, char*** argv) {
    std::vector<machina::Flag> flags = Flags();

    // Parse known flags
    string usage = machina::Flags::Usage(
        absl::StrCat(*argv[0], " Op1 [Op2 ...]"), flags);
    QCHECK(machina::Flags::Parse(argc, *argv, flags)) << usage;  // Crash OK

    // Initialize any TensorFlow support, parsing boilerplate flags (e.g. logs)
    machina::port::InitMain(usage.c_str(), argc, argv);

    // Validate flags
    if (help_) {
      LOG(QFATAL) << usage;  // Crash OK
    }

    QCHECK(!source_dir_.empty()) << usage;  // Crash OK
    QCHECK(!output_dir_.empty()) << usage;  // Crash OK
    QCHECK(!category_.empty()) << usage;    // Crash OK

    // Remaining arguments (i.e. the positional args) are the requested Op names
    op_names_.assign((*argv) + 1, (*argv) + (*argc));
  }

  generator::cpp::CppConfig CppConfig() {
    return generator::cpp::CppConfig(category_);
  }

  generator::PathConfig PathConfig() {
    return generator::PathConfig(output_dir_, source_dir_, api_dirs_,
                                 op_names_);
  }

 private:
  std::vector<machina::Flag> Flags() {
    return {
        machina::Flag("help", &help_, "Print this help message."),
        machina::Flag("category", &category_,
                         "Category for generated ops (e.g. 'math', 'array')."),
        machina::Flag(
            "namespace", &name_space_,
            "Compact C++ namespace, default is 'machina::ops'."),
        machina::Flag(
            "output_dir", &output_dir_,
            "Directory into which output files will be generated."),
        machina::Flag(
            "source_dir", &source_dir_,
            "The machina root directory, e.g. 'machina/' for "
            "in-source include paths. Any path underneath the "
            "machina root is also accepted."),
        machina::Flag(
            "api_dirs", &api_dirs_,
            "Comma-separated list of directories containing API definitions.")};
  }

  bool help_ = false;
  string category_;
  string name_space_;
  string output_dir_;
  string source_dir_;
  string api_dirs_;
  std::vector<string> op_names_;
};

}  // namespace

int main(int argc, char* argv[]) {
  MainConfig config;
  config.InitMain(&argc, &argv);
  generator::CppGenerator generator(config.CppConfig(), config.PathConfig());
  generator.WriteHeaderFile();
  generator.WriteSourceFile();
  return 0;
}
