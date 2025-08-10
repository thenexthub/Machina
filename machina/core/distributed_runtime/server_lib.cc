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

#include "machina/core/distributed_runtime/server_lib.h"

#include <unordered_map>

#include "machina/core/lib/core/errors.h"
#include "machina/core/platform/mutex.h"

namespace machina {

namespace {
mutex* get_server_factory_lock() {
  static mutex server_factory_lock(LINKER_INITIALIZED);
  return &server_factory_lock;
}

typedef std::unordered_map<string, ServerFactory*> ServerFactories;
ServerFactories* server_factories() {
  static ServerFactories* factories = new ServerFactories;
  return factories;
}
}  // namespace

/* static */
void ServerFactory::Register(const string& server_type,
                             ServerFactory* factory) {
  mutex_lock l(*get_server_factory_lock());
  if (!server_factories()->insert({server_type, factory}).second) {
    LOG(ERROR) << "Two server factories are being registered under "
               << server_type;
  }
}

/* static */
absl::Status ServerFactory::GetFactory(const ServerDef& server_def,
                                       ServerFactory** out_factory) {
  mutex_lock l(*get_server_factory_lock());
  for (const auto& server_factory : *server_factories()) {
    if (server_factory.second->AcceptsOptions(server_def)) {
      *out_factory = server_factory.second;
      return absl::OkStatus();
    }
  }

  std::vector<string> server_names;
  for (const auto& server_factory : *server_factories()) {
    server_names.push_back(server_factory.first);
  }

  return errors::NotFound(
      "No server factory registered for the given ServerDef: ",
      server_def.DebugString(), "\nThe available server factories are: [ ",
      absl::StrJoin(server_names, ", "), " ]");
}

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
absl::Status NewServer(const ServerDef& server_def,
                       std::unique_ptr<ServerInterface>* out_server) {
  ServerFactory* factory;
  TF_RETURN_IF_ERROR(ServerFactory::GetFactory(server_def, &factory));
  return factory->NewServer(server_def, ServerFactory::Options(), out_server);
}

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
absl::Status NewServerWithOptions(
    const ServerDef& server_def, const ServerFactory::Options& options,
    std::unique_ptr<ServerInterface>* out_server) {
  ServerFactory* factory;
  TF_RETURN_IF_ERROR(ServerFactory::GetFactory(server_def, &factory));
  return factory->NewServer(server_def, options, out_server);
}

}  // namespace machina
