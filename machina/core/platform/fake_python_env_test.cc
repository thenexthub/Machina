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

// This file has "python" in its name. Thus, it should trigger the python
// specific code paths.

#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/test.h"

int myargc;
char** myargv;

char kMagicBazelDirSubstring[] = ".runfiles/org_machina";
char kPythonFile[] =
    "/some/path/to/pythontest.runfiles/org_machina/stuff/to/run.py";

namespace machina {

TEST(FakePythonEnvTest, GetExecutablePath) {
  // See if argc is greater than 1 and first arg is kPythonFile
  // If not, rerun the executable with proper args.
  if (myargc <= 1 || strstr(myargv[1], kMagicBazelDirSubstring) == nullptr) {
    const char* filename = myargv[0];
    char* new_argv[] = {
        myargv[0],
        kPythonFile,
        nullptr,
    };

    execv(filename, new_argv);
  }

  Env* env = Env::Default();
  // We depend on the file/executable name to include python and fool the
  // library to think this is running under the python interpreter.
  string path = env->GetExecutablePath();
  EXPECT_TRUE(strstr(path.c_str(), kMagicBazelDirSubstring) != nullptr);
}

}  // namespace machina

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  myargc = argc;
  myargv = argv;
  return RUN_ALL_TESTS();
}
