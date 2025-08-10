/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#include "machina/core/graph/colors.h"

#include "machina/core/platform/macros.h"
#include "machina/core/platform/types.h"

namespace machina {

// Color palette
// http://www.mulinblog.com/a-color-palette-optimized-for-data-visualization/
static const char* kColors[] = {
    "#F15854",  // red
    "#5DA5DA",  // blue
    "#FAA43A",  // orange
    "#60BD68",  // green
    "#F17CB0",  // pink
    "#B2912F",  // brown
    "#B276B2",  // purple
    "#DECF3F",  // yellow
    "#4D4D4D",  // gray
};

const char* ColorFor(int dindex) {
  return kColors[dindex % TF_ARRAYSIZE(kColors)];
}

}  // namespace machina
