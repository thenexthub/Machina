/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include "toolchain/FileCheck/FileCheck.h"
#include "toolchain/Support/SourceMgr.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "machina/python/lib/core/pybind11_lib.h"
#include "machina/python/lib/core/pybind11_status.h"

PYBIND11_MODULE(filecheck_wrapper, m) {
  m.def("check", [](std::string input, std::string check) {
    toolchain::FileCheckRequest fcr;
    toolchain::FileCheck fc(fcr);
    toolchain::SourceMgr SM = toolchain::SourceMgr();
    SM.AddNewSourceBuffer(toolchain::MemoryBuffer::getMemBuffer(input),
                          toolchain::SMLoc());
    SM.AddNewSourceBuffer(toolchain::MemoryBuffer::getMemBuffer(check),
                          toolchain::SMLoc());
    fc.readCheckFile(SM, toolchain::StringRef(check));
    return fc.checkInput(SM, toolchain::StringRef(input));
  });
}
