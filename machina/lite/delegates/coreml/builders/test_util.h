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
#ifndef MACHINA_LITE_DELEGATES_COREML_BUILDERS_TEST_UTIL_H_
#define MACHINA_LITE_DELEGATES_COREML_BUILDERS_TEST_UTIL_H_

#include "machina/lite/delegates/coreml/coreml_delegate.h"
#include "machina/lite/kernels/test_util.h"

#import <XCTest/XCTest.h>

namespace tflite {
namespace delegates {
namespace coreml {
class SingleOpModelWithCoreMlDelegate : public tflite::SingleOpModel {
 public:
  static const char kDelegateName[];

  SingleOpModelWithCoreMlDelegate();
  tflite::Interpreter* interpreter() { return interpreter_.get(); }

 protected:
  using SingleOpModel::builder_;

 private:
  tflite::Interpreter::TfLiteDelegatePtr delegate_;
  TfLiteCoreMlDelegateOptions params_ = {
      .enabled_devices = TfLiteCoreMlDelegateAllDevices,
      .min_nodes_per_partition = 1,
  };
};

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

@interface BaseOpTest : XCTestCase
@property tflite::delegates::coreml::SingleOpModelWithCoreMlDelegate* model;
- (void)validateInterpreter:(tflite::Interpreter*)interpreter;
- (void)checkInterpreterNotDelegated:(tflite::Interpreter*)interpreter;
- (void)invokeAndValidate;
- (void)invokeAndCheckNotDelegated;
@end

#endif  // MACHINA_LITE_DELEGATES_COREML_BUILDERS_TEST_UTIL_H_
