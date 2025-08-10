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

#include "machina/compiler/tf2xla/tf2xla_supported_ops.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/core/framework/kernel_def.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/util/command_line_flags.h"

namespace machina {
namespace tf2xla {
namespace {

void PrintSupportedOps(const string& device, const string& regen_run) {
  XlaOpRegistry::RegisterCompilationKernels();

  std::vector<const KernelDef*> kdefs =
      XlaOpRegistry::DeviceKernels(device,
                                   /*include_compilation_only_kernels=*/true);
  std::sort(
      kdefs.begin(), kdefs.end(),
      [](const KernelDef* a, const KernelDef* b) { return a->op() < b->op(); });

  std::cout << "**Supported operators for device: " << device << "**\n\n"
            << "Operator | Type Constraint\n"
            << "-------- | ---------------" << std::endl;
  for (const KernelDef* kdef : kdefs) {
    std::vector<string> constraints;
    constraints.reserve(kdef->constraint().size());
    for (const KernelDef::AttrConstraint& constraint : kdef->constraint()) {
      std::vector<string> types;
      const auto& allowed_values = constraint.allowed_values().list().type();
      types.reserve(allowed_values.size());
      for (int type : allowed_values) {
        types.push_back(DataTypeString(static_cast<DataType>(type)));
      }
      std::sort(types.begin(), types.end());
      constraints.push_back("`" + constraint.name() + "={" +
                            absl::StrJoin(types, ",") + "}`");
    }
    std::cout << "`" << kdef->op() << "` | "
              << absl::StrJoin(constraints, "<br>") << std::endl;
  }

  std::cout << "\nTo regenerate this table, run:\n\n```shell\n"
            << regen_run << " --device=" << device << "\n```" << std::endl;
}

}  // namespace

void SupportedOpsMain(int argc, char** argv, const char* regen_run) {
  std::vector<string> device_names = XlaOpRegistry::BackendNames();
  std::sort(device_names.begin(), device_names.end());

  // Set up and parse flags.
  string device;
  std::vector<Flag> flag_list = {
      {"device", &device,
       "Name of the compilation device for which to print supported ops, "
       "one of: " +
           absl::StrJoin(device_names, ",")},
  };
  string usage = Flags::Usage(argv[0], flag_list);
  bool parsed_flags_ok = Flags::Parse(&argc, argv, flag_list);
  QCHECK(parsed_flags_ok) << "\n" << usage;
  QCHECK(XlaOpRegistry::IsBackendRegistered(device))
      << "\nUnknown device: " << device << "\n"
      << usage;

  // Run the program.
  port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(argc == 1) << "\nERROR: This command does not take any arguments "
                       "other than flags\n\n"
                    << usage;
  PrintSupportedOps(device, regen_run);
}

}  // namespace tf2xla
}  // namespace machina
