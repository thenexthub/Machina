###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, July 14, 2025.                                              #
#                                                                             #
#   Licensed under the Apache License, Version 2.0 (the "License");           #
#   you may not use this file except in compliance with the License.          #
#   You may obtain a copy of the License at:                                  #
#                                                                             #
#       http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                             #
#   Unless required by applicable law or agreed to in writing, software       #
#   distributed under the License is distributed on an "AS IS" BASIS,         #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
#   See the License for the specific language governing permissions and       #
#   limitations under the License.                                            #
#                                                                             #
#   Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,            #
#   Middletown, DE 19709, New Castle County, USA.                             #
#                                                                             #
###############################################################################
"""Tool to re-align the tflite flatbuffer via the C++ flatbuffer api."""

from absl import app

from tflite_micro.machina.lite.micro.tools import tflite_flatbuffer_align_wrapper


def main(argv):
  try:
    input_model_path = argv[1]
    output_model_path = argv[2]
  except IndexError:
    print('usage: ', argv[0], ' <input tflite> <output tflite>\n')
  else:
    tflite_flatbuffer_align_wrapper.align_tflite_model(input_model_path,
                                                       output_model_path)


if __name__ == '__main__':
  app.run(main)
