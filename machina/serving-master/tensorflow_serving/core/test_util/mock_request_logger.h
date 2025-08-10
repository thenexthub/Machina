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

#ifndef MACHINA_SERVING_CORE_TEST_UTIL_MOCK_REQUEST_LOGGER_H_
#define MACHINA_SERVING_CORE_TEST_UTIL_MOCK_REQUEST_LOGGER_H_

#include <vector>

#include "google/protobuf/message.h"
#include <gmock/gmock.h>
#include "machina/core/lib/core/status.h"
#include "machina_serving/apis/logging.pb.h"
#include "machina_serving/config/logging_config.pb.h"
#include "machina_serving/core/log_collector.h"
#include "machina_serving/core/request_logger.h"

namespace machina {
namespace serving {

class MockRequestLogger : public RequestLogger {
 public:
  // Unfortunately NiceMock doesn't support ctors with move-only types, so we
  // have to do this workaround.
  MockRequestLogger(const LoggingConfig& logging_config,
                    const std::vector<string>& saved_model_tags,
                    LogCollector* log_collector,
                    std::function<void(void)> notify_destruction =
                        std::function<void(void)>())
      : RequestLogger(logging_config, saved_model_tags,
                      std::unique_ptr<LogCollector>(log_collector)),
        notify_destruction_(std::move(notify_destruction)) {}

  virtual ~MockRequestLogger() {
    if (notify_destruction_) {
      notify_destruction_();
    }
  }

  MOCK_METHOD(Status, CreateLogMessage,
              (const google::protobuf::Message& request, const google::protobuf::Message& response,
               const LogMetadata& log_metadata,
               std::unique_ptr<google::protobuf::Message>* log),
              (override));

  MOCK_METHOD(LogMetadata, FillLogMetadata, (const LogMetadata&), (override));

 private:
  std::function<void(void)> notify_destruction_;
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_TEST_UTIL_MOCK_REQUEST_LOGGER_H_
