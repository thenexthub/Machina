/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#include "machina/compiler/xla/xla_client/tf_logging.h"

namespace xla {
namespace internal {

void ErrorGenerator::operator&(const std::basic_ostream<char>& oss) const {
  const ErrorSink& sink = dynamic_cast<const ErrorSink&>(oss);
  auto sink_str = sink.str();
  TF_LOG(ERROR) << sink_str;
  std::stringstream ess;
  ess << file_ << ":" << line_ << " : " << sink_str;
  TF_LOG(FATAL) << ess.str();
}

}  // namespace internal
}  // namespace xla
