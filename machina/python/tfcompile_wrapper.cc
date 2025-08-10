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

#include <string>

#ifdef __s390x__
#include "toolchain/ADT/StringRef.h"
#include "toolchain/TargetParser/Host.h"
#endif
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "machina/compiler/aot/compile.h"
#include "machina/compiler/aot/flags.h"
#include "machina/python/lib/core/pybind11_lib.h"
#include "machina/python/lib/core/pybind11_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_tfcompile, m) {
  m.doc() = R"pbdoc(
    _pywrap_tfcompile
    -----
  )pbdoc";

  m.def(
      "Compile",
      [](std::string graph, std::string config, std::string target_triple,
         std::string target_cpu, std::string target_features,
         std::string entry_point, std::string cpp_class,
         std::string out_function_object, std::string out_metadata_object,
         std::string out_header, std::string out_session_module,
         std::string out_constant_buffers_object, std::string mlir_components,
         bool gen_name_to_index, bool gen_program_shape) {
        machina::tfcompile::MainFlags flags;
        flags.graph = std::move(graph);
        flags.config = std::move(config);
#ifdef __s390x__
        flags.target_triple = std::move(
            target_triple.empty() ? toolchain::sys::getDefaultTargetTriple()
                                  : target_triple);
        flags.target_cpu =
            std::move(target_cpu.empty() ? toolchain::sys::getHostCPUName().str()
                                         : target_cpu);
#else
        flags.target_triple = std::move(target_triple);
        flags.target_cpu = std::move(target_cpu);
#endif
        flags.target_features = std::move(target_features);
        flags.entry_point = std::move(entry_point);
        flags.cpp_class = std::move(cpp_class);
        flags.out_function_object = std::move(out_function_object);
        flags.out_metadata_object = std::move(out_metadata_object);
        flags.out_header = std::move(out_header);
        flags.out_session_module = std::move(out_session_module);
        flags.out_constant_buffers_object =
            std::move(out_constant_buffers_object);
        flags.mlir_components = std::move(mlir_components);

        // C++ codegen options
        flags.gen_name_to_index = gen_name_to_index;
        flags.gen_program_shape = gen_program_shape;

        machina::MaybeRaiseFromStatus(machina::tfcompile::Main(flags));
      },
      py::arg("graph") = "", py::arg("config") = "",
#ifdef __s390x__
      py::arg("target_triple") = "",
#else
      py::arg("target_triple") = "x86_64-pc-linux",
#endif
      py::arg("target_cpu") = "", py::arg("target_features") = "",
      py::arg("entry_point") = "entry", py::arg("cpp_class") = "",
      py::arg("out_function_object") = "out_model.o",
      py::arg("out_metadata_object") = "out_helper.o",
      py::arg("out_header") = "out.h", py::arg("out_session_module") = "",
      py::arg("out_constant_buffers_object") = "",
      py::arg("mlir_components") = "", py::arg("gen_name_to_index") = false,
      py::arg("gen_program_shape") = false);
}
