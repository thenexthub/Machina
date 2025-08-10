/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

#if defined(WIN32) || defined(_WIN32)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif

#define TOKEN_CONCAT(a, b) a##b
#define WRAPPED_PY_MODULE(name)                                  \
  extern "C" void *TOKEN_CONCAT(Wrapped_PyInit_, name)();        \
  extern "C" EXPORT_SYMBOL void *TOKEN_CONCAT(PyInit_, name)() { \
    return TOKEN_CONCAT(Wrapped_PyInit_, name)();                \
  }

WRAPPED_PY_MODULE(WRAPPED_PY_MODULE_NAME)
