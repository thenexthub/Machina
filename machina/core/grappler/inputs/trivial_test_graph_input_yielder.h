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

#ifndef MACHINA_CORE_GRAPPLER_INPUTS_TRIVIAL_TEST_GRAPH_INPUT_YIELDER_H_
#define MACHINA_CORE_GRAPPLER_INPUTS_TRIVIAL_TEST_GRAPH_INPUT_YIELDER_H_

#include <string>
#include <vector>
#include "machina/core/grappler/inputs/input_yielder.h"

namespace machina {
namespace grappler {

class Cluster;
struct GrapplerItem;

class TrivialTestGraphInputYielder : public InputYielder {
 public:
  TrivialTestGraphInputYielder(int num_stages, int width, int tensor_size,
                               bool insert_queue,
                               const std::vector<std::string>& device_names);
  bool NextItem(GrapplerItem* item) override;

 private:
  const int num_stages_;
  const int width_;
  const int tensor_size_;
  const bool insert_queue_;
  std::vector<std::string> device_names_;
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_INPUTS_TRIVIAL_TEST_GRAPH_INPUT_YIELDER_H_
