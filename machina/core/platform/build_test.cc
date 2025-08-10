/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/core/platform/abi.h"
#include "machina/core/platform/byte_order.h"
#include "machina/core/platform/cord.h"
#include "machina/core/platform/cpu_feature_guard.h"
#include "machina/core/platform/cpu_info.h"
#include "machina/core/platform/demangle.h"
#include "machina/core/platform/denormal.h"
#include "machina/core/platform/dynamic_annotations.h"
#include "machina/core/platform/env_time.h"
#include "machina/core/platform/file_statistics.h"
#include "machina/core/platform/fingerprint.h"
#include "machina/core/platform/gif.h"
#include "machina/core/platform/host_info.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/platform/jpeg.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mem.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/net.h"
#include "machina/core/platform/numa.h"
#include "machina/core/platform/numbers.h"
#include "machina/core/platform/platform.h"
#include "machina/core/platform/platform_strings.h"
#include "machina/core/platform/png.h"
#include "machina/core/platform/prefetch.h"
#include "machina/core/platform/protobuf.h"
#if !defined(__ANDROID__)
#include "machina/core/platform/rocm_rocdl_path.h"
#endif
#include "machina/core/platform/scanner.h"
#include "machina/core/platform/setround.h"
#include "machina/core/platform/snappy.h"
#include "machina/core/platform/stacktrace.h"
#include "machina/core/platform/stacktrace_handler.h"
#include "machina/core/platform/str_util.h"
#include "machina/core/platform/strcat.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/stringprintf.h"
#include "machina/core/platform/subprocess.h"
#include "machina/core/platform/tensor_coding.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/thread_annotations.h"
#include "machina/core/platform/threadpool_interface.h"
#include "machina/core/platform/threadpool_options.h"
#include "machina/core/platform/tstring.h"
#include "machina/core/platform/types.h"

int main(int argc, char *argv[]) { return 0; }
