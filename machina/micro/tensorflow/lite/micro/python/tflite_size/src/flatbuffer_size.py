###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Saturday, May 31, 2025.                                             #
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
###############################################################################=
import json
import sys

from tflite_micro.machina.lite.micro.python.tflite_size.src import flatbuffer_size_wrapper_pybind
from tflite_micro.machina.lite.micro.python.tflite_size.src import flatbuffer_size_graph
from tflite_micro.machina.lite.micro.python.tflite_size.src import flatbuffer_size_graph_html_converter


def convert_tflite_to_html(in_flatbuf):
  """ Given a input tflite flatbuffer, returns a html and a json with size info"""
  size_wrapper = flatbuffer_size_wrapper_pybind.FlatbufferSize()
  out_json = size_wrapper.convertToJsonString(in_flatbuf)
  json_as_dict = json.loads(out_json)

  formatted_json = json.dumps(json_as_dict)

  graph_builder = flatbuffer_size_graph.FlatbufferSizeGraph()
  graph_builder.create_graph(json_as_dict)

  html_converter = flatbuffer_size_graph_html_converter.HtmlConverter()
  html_string = graph_builder.display_graph(html_converter)

  return html_string, formatted_json


def main(argv):
  try:
    tflite_input = argv[1]
    html_output = argv[2]
  except IndexError:
    print("Usage: %s <input tflite> <output html>" % (argv[0]))
  else:
    with open(tflite_input, 'rb') as f:
      in_flatbuf = f.read()

    html_string = convert_tflite_to_html(in_flatbuf)[0]

    with open(html_output, 'w') as f:
      f.write(html_string)


if __name__ == '__main__':
  main(sys.argv)
