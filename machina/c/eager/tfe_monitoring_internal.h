/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_C_EAGER_TFE_MONITORING_INTERNAL_H_
#define MACHINA_C_EAGER_TFE_MONITORING_INTERNAL_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "machina/core/lib/monitoring/counter.h"
#include "machina/core/lib/monitoring/gauge.h"
#include "machina/core/lib/monitoring/sampler.h"
#include "machina/core/platform/types.h"

struct TFE_MonitoringCounterCell {
  machina::monitoring::CounterCell cell;
};

template <int NumLabels>
struct TFE_MonitoringCounter {
  template <typename... LabelDesc>
  TFE_MonitoringCounter(const char* name, const char* description,
                        LabelDesc&&... label) {
    counter = absl::WrapUnique(machina::monitoring::Counter<NumLabels>::New(
        name, description, label...));
  }

  std::unique_ptr<machina::monitoring::Counter<NumLabels>> counter;
};

struct TFE_MonitoringCounter0 : TFE_MonitoringCounter<0> {
  using TFE_MonitoringCounter::TFE_MonitoringCounter;
};
struct TFE_MonitoringCounter1 : TFE_MonitoringCounter<1> {
  using TFE_MonitoringCounter::TFE_MonitoringCounter;
};
struct TFE_MonitoringCounter2 : TFE_MonitoringCounter<2> {
  using TFE_MonitoringCounter::TFE_MonitoringCounter;
};

struct TFE_MonitoringIntGaugeCell {
  machina::monitoring::GaugeCell<int64_t> cell;
};
struct TFE_MonitoringStringGaugeCell {
  machina::monitoring::GaugeCell<machina::string> cell;
};
struct TFE_MonitoringBoolGaugeCell {
  machina::monitoring::GaugeCell<bool> cell;
};

template <typename ValueType, int NumLabels>
struct TFE_MonitoringGauge {
  template <typename... LabelDesc>
  TFE_MonitoringGauge(const char* name, const char* description,
                      LabelDesc&&... label) {
    gauge = absl::WrapUnique(
        machina::monitoring::Gauge<ValueType, NumLabels>::New(
            name, description, label...));
  }

  std::unique_ptr<machina::monitoring::Gauge<ValueType, NumLabels>> gauge;
};

struct TFE_MonitoringIntGauge0 : TFE_MonitoringGauge<int64_t, 0> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringIntGauge1 : TFE_MonitoringGauge<int64_t, 1> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringIntGauge2 : TFE_MonitoringGauge<int64_t, 2> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};

struct TFE_MonitoringStringGauge0 : TFE_MonitoringGauge<machina::string, 0> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringStringGauge1 : TFE_MonitoringGauge<machina::string, 1> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringStringGauge2 : TFE_MonitoringGauge<machina::string, 2> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringStringGauge3 : TFE_MonitoringGauge<machina::string, 3> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringStringGauge4 : TFE_MonitoringGauge<machina::string, 4> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};

struct TFE_MonitoringBoolGauge0 : TFE_MonitoringGauge<bool, 0> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringBoolGauge1 : TFE_MonitoringGauge<bool, 1> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};
struct TFE_MonitoringBoolGauge2 : TFE_MonitoringGauge<bool, 2> {
  using TFE_MonitoringGauge::TFE_MonitoringGauge;
};

struct TFE_MonitoringBuckets {
  explicit TFE_MonitoringBuckets(
      std::function<std::unique_ptr<machina::monitoring::Buckets>(void)>
          fn) {
    create_buckets = fn;
  }

  std::function<std::unique_ptr<machina::monitoring::Buckets>(void)>
      create_buckets;
};

struct TFE_MonitoringSamplerCell {
  machina::monitoring::SamplerCell cell;
};

template <int NumLabels>
struct TFE_MonitoringSampler {
  template <typename... LabelDesc>
  TFE_MonitoringSampler(
      const char* name,
      std::unique_ptr<machina::monitoring::Buckets> buckets,
      const char* description, LabelDesc&&... label) {
    sampler = absl::WrapUnique(machina::monitoring::Sampler<NumLabels>::New(
        {name, description, label...}, std::move(buckets)));
  }

  std::unique_ptr<machina::monitoring::Sampler<NumLabels>> sampler;
};

struct TFE_MonitoringSampler0 : TFE_MonitoringSampler<0> {
  using TFE_MonitoringSampler::TFE_MonitoringSampler;
};
struct TFE_MonitoringSampler1 : TFE_MonitoringSampler<1> {
  using TFE_MonitoringSampler::TFE_MonitoringSampler;
};
struct TFE_MonitoringSampler2 : TFE_MonitoringSampler<2> {
  using TFE_MonitoringSampler::TFE_MonitoringSampler;
};

#endif  // MACHINA_C_EAGER_TFE_MONITORING_INTERNAL_H_
