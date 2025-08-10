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
#ifndef MACHINA_LITE_MICRO_COMPATIBILITY_H_
#define MACHINA_LITE_MICRO_COMPATIBILITY_H_

// C++ will automatically create class-specific delete operators for virtual
// objects, which by default call the global delete function. For embedded
// applications we want to avoid this, and won't be calling new/delete on these
// objects, so we need to override the default implementation with one that does
// nothing to avoid linking in ::delete().
// This macro needs to be included in all subclasses of a virtual base class in
// the private section.
#ifdef TF_LITE_STATIC_MEMORY
#define TF_LITE_REMOVE_VIRTUAL_DELETE \
  void operator delete(void* p) {}
#else
#define TF_LITE_REMOVE_VIRTUAL_DELETE
#endif

#endif  // MACHINA_LITE_MICRO_COMPATIBILITY_H_
