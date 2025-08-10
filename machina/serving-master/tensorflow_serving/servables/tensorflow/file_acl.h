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

#ifndef THIRD_PARTY_MACHINA_SERVING_SERVABLES_MACHINA_FILE_ACL_H_
#define THIRD_PARTY_MACHINA_SERVING_SERVABLES_MACHINA_FILE_ACL_H_

#include "machina_serving/util/oss_or_google.h"

#ifdef MACHINA_SERVING_GOOGLE
#include "machina_serving/servables/machina/google/file_acl.h"
#else
#include "machina_serving/servables/machina/oss/file_acl.h"
#endif

#endif  // THIRD_PARTY_MACHINA_SERVING_SERVABLES_MACHINA_FILE_ACL_H_
