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
#include <utility>

#include "machina/compiler/mlir/quantization/machina/exported_model.pb.h"
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"
#include "machina/xla/tsl/protobuf/coordination_service.pb.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/device_attributes.pb.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/grappler/costs/op_performance_data.pb.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina/core/protobuf/data_service.pb.h"
#include "machina/core/protobuf/device_properties.pb.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/core/protobuf/service_config.pb.h"
#include "machina/dtensor/proto/layout.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace machina {
namespace python {
void protobuf_inline_symbols_enforcer() {
  machina::NamedDevice named_device;
  named_device.mutable_properties();
  named_device.properties();

  machina::NamedDevice named_device_move(std::move(named_device));
  named_device_move.mutable_properties();

  machina::quantization::ExportedModel exported_model;
  exported_model.function_aliases();

  machina::profiler::XSpace x_space;
  x_space.mutable_hostnames();
  x_space.mutable_hostnames(0);

  machina::dtensor::LayoutProto layout_proto;
  layout_proto.GetDescriptor();
  layout_proto.GetReflection();
  layout_proto.default_instance();

  machina::dtensor::MeshProto mesh_proto;
  mesh_proto.GetDescriptor();
  mesh_proto.GetReflection();
  mesh_proto.default_instance();

  machina::FunctionDef function_def;
  function_def.descriptor();
  function_def.GetDescriptor();
  function_def.GetReflection();
  function_def.default_instance();

  machina::FunctionDefLibrary function_def_library;
  function_def_library.descriptor();

  machina::GraphDef graph_def;
  graph_def.descriptor();
  graph_def.GetDescriptor();
  graph_def.GetReflection();
  graph_def.default_instance();

  machina::MetaGraphDef meta_graph_def;
  meta_graph_def.GetDescriptor();
  meta_graph_def.GetReflection();
  meta_graph_def.default_instance();

  machina::AttrValue attr_value;
  attr_value.default_instance();
  machina::AttrValue_ListValue list_value;
  list_value.add_b(false);

  OpPerformanceList performance_list;

  machina::ConfigProto config_proto;
  config_proto.default_instance();

  machina::data::experimental::DispatcherConfig dispatcher_config;
  dispatcher_config.default_instance();

  machina::data::experimental::WorkerConfig worker_config;
  worker_config.default_instance();

  machina::data::DataServiceMetadata data_service_metadata;
  machina::quantization::QuantizationOptions quantization_options;
  machina::CoordinatedTask coordinated_task;
  machina::DeviceAttributes device_attributes;
}
}  // namespace python
}  // namespace machina
