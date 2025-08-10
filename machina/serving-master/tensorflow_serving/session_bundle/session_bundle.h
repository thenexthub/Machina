/* Copyright 2019 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MACHINA_SERVING_SESSION_BUNDLE_SESSION_BUNDLE_H_
#define MACHINA_SERVING_SESSION_BUNDLE_SESSION_BUNDLE_H_

#include "machina_serving/util/oss_or_google.h"

#ifdef MACHINA_SERVING_GOOGLE
#include "machina/contrib/session_bundle/session_bundle.h"
#else
#include "machina_serving/session_bundle/oss/session_bundle.h"
#endif

#endif  // MACHINA_SERVING_SESSION_BUNDLE_SESSION_BUNDLE_H_
