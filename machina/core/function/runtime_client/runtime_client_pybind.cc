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

#include "absl/status/status.h"
#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/core/common_runtime/eager/context.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/function/runtime_client/runtime_client.h"
#include "machina/python/lib/core/pybind11_status.h"

PYBIND11_MAKE_OPAQUE(machina::EagerContext);

PYBIND11_MODULE(runtime_client_pybind, m) {
  pybind11::class_<machina::EagerContext, machina::EagerContextPtr>
      EagerContext(m, "EagerContext");
  pybind11::class_<absl::Status> Status(m, "Status", pybind11::module_local());

  m.def("GlobalEagerContext", &machina::core::function::GlobalEagerContext,
        pybind11::return_value_policy::reference);

  m.def("GlobalPythonEagerContext",
        &machina::core::function::GlobalPythonEagerContext,
        pybind11::return_value_policy::reference);

  pybind11::class_<machina::core::function::Runtime> runtime(m, "Runtime");

  pybind11::enum_<machina::core::function::Runtime::Dialect>(runtime,
                                                                "Dialect")
      .value("TFG", machina::core::function::Runtime::Dialect::TFG)
      .value("TF", machina::core::function::Runtime::Dialect::TF);

  runtime.def(pybind11::init<machina::EagerContext&>());
  // TODO(mdan): Rename to GetFunctionProto once pybind11_protobuf available
  runtime.def(
      "GetFunctionProtoString",
      [](machina::core::function::Runtime& r, const std::string& name) {
        return pybind11::bytes(r.GetFunctionProto(name)->SerializeAsString());
      },
      pybind11::return_value_policy::reference);
  // TODO(mdan): Rename to CreateFunction once pybind11_protobuf available
  runtime.def(
      "CreateFunctionFromString",
      [](machina::core::function::Runtime& r, const std::string& def) {
        machina::FunctionDef proto;
        proto.ParseFromString(def);
        return r.CreateFunction(proto);
      });
  runtime.def("TransformFunction",
              &machina::core::function::Runtime::TransformFunction,
              pybind11::arg("name"), pybind11::arg("pipeline_name"),
              pybind11::arg("dialect") =
                  machina::core::function::Runtime::Dialect::TFG);
}
