/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include "machina/xla/tsl/platform/cloud/compute_engine_metadata_client.h"

#include <cstdlib>
#include <utility>

#include "absl/strings/str_cat.h"
#include "machina/xla/tsl/platform/cloud/curl_http_request.h"

namespace tsl {

namespace {

// The environment variable to override the compute engine metadata endpoint.
constexpr char kGceMetadataHost[] = "GCE_METADATA_HOST";

// The URL to retrieve metadata when running in Google Compute Engine.
constexpr char kGceMetadataBaseUrl[] =
    "http://metadata.google.internal/computeMetadata/v1/";

}  // namespace

ComputeEngineMetadataClient::ComputeEngineMetadataClient(
    std::shared_ptr<HttpRequest::Factory> http_request_factory,
    const RetryConfig& config)
    : http_request_factory_(std::move(http_request_factory)),
      retry_config_(config) {}

absl::Status ComputeEngineMetadataClient::GetMetadata(
    const string& path, std::vector<char>* response_buffer) {
  const auto get_metadata_from_gce = [path, response_buffer, this]() {
    string metadata_url;
    const char* metadata_url_override = std::getenv(kGceMetadataHost);
    if (metadata_url_override) {
      metadata_url = absl::StrCat("http://", metadata_url_override,
                                  "/computeMetadata/v1/");
    } else {
      metadata_url = kGceMetadataBaseUrl;
    }
    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    request->SetUri(metadata_url + path);
    request->AddHeader("Metadata-Flavor", "Google");
    request->SetResultBuffer(response_buffer);
    TF_RETURN_IF_ERROR(request->Send());
    return absl::OkStatus();
  };

  return RetryingUtils::CallWithRetries(get_metadata_from_gce, retry_config_);
}

}  // namespace tsl
