/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IRReader/IRReader.h"
#include "toolchain/Support/SourceMgr.h"
#include "tsl/platform/logging.h"

namespace {

static void DieWithSMDiagnosticError(toolchain::SMDiagnostic* diagnostic) {
  LOG(FATAL) << diagnostic->getFilename().str() << ":"
             << diagnostic->getLineNo() << ":" << diagnostic->getColumnNo()
             << ": " << diagnostic->getMessage().str();
}

}  // namespace

namespace xla::gpu {

std::unique_ptr<toolchain::Module> LoadIRModule(const std::string& filename,
                                           toolchain::LLVMContext* toolchain_context) {
  toolchain::SMDiagnostic diagnostic_err;
  std::unique_ptr<toolchain::Module> module =
      toolchain::getLazyIRFileModule(filename, diagnostic_err, *toolchain_context,
                                /*ShouldLazyLoadMetadata=*/true);

  if (module == nullptr) {
    DieWithSMDiagnosticError(&diagnostic_err);
  }

  return module;
}

}  // namespace xla::gpu
