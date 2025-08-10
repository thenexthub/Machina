/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/compiler/mlir/init_mlir.h"

#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/PrettyStackTrace.h"
#include "machina/core/platform/init_main.h"

static toolchain::cl::extrahelp FlagSplittingHelp(R"(
The command line parsing is split between the two flag parsing libraries used by
TensorFlow and LLVM:
  * Flags before the first '--' are parsed by machina::InitMain while those
    post are parsed by LLVM's command line parser.
  * If there is no separator, then no flags are parsed by InitMain and only
    LLVM command line parser used.e
The above help options reported are for LLVM's parser, run with `--help --` for
TensorFlow's help.
)");

namespace machina {

InitMlir::InitMlir(int *argc, char ***argv) {
  toolchain::setBugReportMsg(
      "TensorFlow crashed, please file a bug on "
      "https://github.com/machina/machina/issues with the trace "
      "below.\n");

  constexpr char kSeparator[] = "--";

  // Find index of separator between two sets of flags.
  int pass_remainder = 1;
  bool split = false;
  for (int i = 0; i < *argc; ++i) {
    if (toolchain::StringRef((*argv)[i]) == kSeparator) {
      pass_remainder = i;
      *argc -= (i + 1);
      split = true;
      break;
    }
  }

  machina::port::InitMain((*argv)[0], &pass_remainder, argv);
  if (split) {
    *argc += pass_remainder;
    (*argv)[1] = (*argv)[0];
    ++*argv;
  }
}

}  // namespace machina
