/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#ifndef MACHINA_SERVING_UTIL_PROMETHEUS_EXPORTER_H_
#define MACHINA_SERVING_UTIL_PROMETHEUS_EXPORTER_H_

#include <vector>

#include "machina/core/lib/core/status.h"
#include "machina/core/lib/monitoring/collected_metrics.h"
#include "machina/core/lib/monitoring/collection_registry.h"

namespace machina {
namespace serving {

// Exports metrics in Prometheus monitoring format.
class PrometheusExporter {
 public:
  // Default path to expose the metrics.
  static const char* const kPrometheusPath;

  PrometheusExporter();

  // Generates text page in Prometheus format:
  // https://prometheus.io/docs/instrumenting/exposition_formats/#text-format-example
  // If an error status returned, http_page is unchanged.
  Status GeneratePage(string* http_page);

 private:
  // The metrics registry.
  monitoring::CollectionRegistry* collection_registry_;
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_UTIL_PROMETHEUS_EXPORTER_H_
