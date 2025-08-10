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

#include <cstddef>
#include <iostream>

#include "machina_serving/util/net_http/internal/net_logging.h"

int main(int argc, char** argv) {
  NET_LOG(INFO, "started!");

  size_t size = 100;
  NET_LOG(ERROR, "read less than specified bytes : %zu", size);

  const char* url = "/url";
  NET_LOG(WARNING, "%s: read less than specified bytes : %zu", url, size);

  NET_LOG(FATAL, "aborted!");

  return 0;  // unexpected
}
