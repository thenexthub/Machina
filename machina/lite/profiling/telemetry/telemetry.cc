/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/lite/profiling/telemetry/telemetry.h"

#include <cstdint>

#include "machina/lite/core/api/profiler.h"
#include "machina/lite/profiling/telemetry/telemetry_status.h"

namespace tflite::telemetry {

void TelemetryReportEvent(TfLiteContext* context, const char* event_name,
                          TfLiteStatus status) {
  if (context->profiler) {
    reinterpret_cast<Profiler*>(context->profiler)
        ->AddEvent(event_name, Profiler::EventType::TELEMETRY_EVENT,
                   TelemetryStatusCode(status).code(),
                   /*event_metadata=*/-1);
  }
}

void TelemetryReportOpEvent(TfLiteContext* context, const char* op_name,
                            int64_t op_index, int64_t subgraph_index,
                            TfLiteStatus status) {
  if (context->profiler) {
    reinterpret_cast<Profiler*>(context->profiler)
        ->AddEvent(op_name, Profiler::EventType::TELEMETRY_EVENT,
                   TelemetryStatusCode(status).code(), op_index,
                   subgraph_index);
  }
}

void TelemetryReportDelegateEvent(TfLiteContext* context,
                                  const char* event_name,
                                  TelemetrySource source, uint32_t code) {
  if (context->profiler) {
    reinterpret_cast<Profiler*>(context->profiler)
        ->AddEvent(event_name, Profiler::EventType::TELEMETRY_DELEGATE_EVENT,
                   TelemetryStatusCode(source, code).code(),
                   /*event_metadata=*/-1);
  }
}

void TelemetryReportDelegateOpEvent(TfLiteContext* context, const char* op_name,
                                    int64_t op_index, int64_t subgraph_index,
                                    TelemetrySource source, uint32_t code) {
  if (context->profiler) {
    reinterpret_cast<Profiler*>(context->profiler)
        ->AddEvent(op_name, Profiler::EventType::TELEMETRY_DELEGATE_EVENT,
                   TelemetryStatusCode(source, code).code(), op_index,
                   subgraph_index);
  }
}

void TelemetryReportSettings(
    TfLiteContext* context, const char* setting_name,
    const TfLiteTelemetryInterpreterSettings* settings) {
  auto* profiler = reinterpret_cast<Profiler*>(context->profiler);
  if (profiler) {
    TfLiteTelemetrySettings telemetry_settings{};
    telemetry_settings.source =
        static_cast<uint32_t>(TelemetrySource::TFLITE_INTERPRETER);
    telemetry_settings.data = reinterpret_cast<const void*>(settings);
    profiler->AddEventWithData(
        setting_name, Profiler::EventType::TELEMETRY_REPORT_SETTINGS,
        reinterpret_cast<const void*>(&telemetry_settings));
  }
}

void TelemetryReportDelegateSettings(TfLiteContext* context,
                                     const char* setting_name,
                                     TelemetrySource source,
                                     const void* settings) {
  auto* profiler = reinterpret_cast<Profiler*>(context->profiler);
  if (profiler) {
    TfLiteTelemetrySettings telemetry_settings{};
    telemetry_settings.source = static_cast<uint32_t>(source);
    telemetry_settings.data = settings;
    profiler->AddEventWithData(
        setting_name, Profiler::EventType::TELEMETRY_DELEGATE_REPORT_SETTINGS,
        reinterpret_cast<const void*>(&telemetry_settings));
  }
}

}  // namespace tflite::telemetry
