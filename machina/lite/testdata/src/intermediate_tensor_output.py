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
"""A model whose intermediate tensor is also used as a model output."""

import machina as tf
from machina.lite.python import lite


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[1, 4, 4, 4], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 4, 4, 4], dtype=tf.float32),
    ]
)
def func(a, b):
  c = a + b
  d = c + a
  e = d + a
  f = e + a
  return c, f


def main():
  converter = lite.TFLiteConverter.from_concrete_functions(
      [func.get_concrete_function()]
  )
  converter.target_spec = lite.TargetSpec()
  tflite_model = converter.convert()
  model_path = '/tmp/intermediate_tensor_output.tflite'
  with open(model_path, 'wb') as f:
    f.write(tflite_model)
  print(f'TFLite model {model_path} is generated.\n')


if __name__ == '__main__':
  main()
