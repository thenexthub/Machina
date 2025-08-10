/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#include "machina/core/api_def/update_api_def.h"

#include "machina/core/framework/op_def.pb.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/lib/io/path.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

TEST(UpdateApiDefTest, TestRemoveDocSingleOp) {
  const string op_def_text = R"opdef(
REGISTER_OP("Op1")
    .Input("a: T")
    .Output("output: T")
    .Attr("b: type")
    .SetShapeFn(shape_inference::UnchangedShape);
)opdef";

  const string op_def_text_with_doc = R"opdef(
REGISTER_OP("Op1")
    .Input("a: T")
    .Output("output: T")
    .Attr("b: type")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Summary for Op1.

Description
for Op1.

b :   Description for b.
a: Description for a.
output: Description for output.
)doc");
)opdef";

  const string op_text = R"(
name: "Op1"
input_arg {
  name: "a"
  description: "Description for a."
}
output_arg {
  name: "output"
  description: "Description for output."
}
attr {
  name: "b"
  description: "Description for b."
}
summary: "Summary for Op1."
description: "Description\nfor Op1."
)";
  OpDef op;
  protobuf::TextFormat::ParseFromString(op_text, &op);  // NOLINT

  EXPECT_EQ(op_def_text,
            RemoveDoc(op, op_def_text_with_doc, 0 /* start_location */));
}

TEST(UpdateApiDefTest, TestRemoveDocMultipleOps) {
  const string op_def_text = R"opdef(
REGISTER_OP("Op1")
    .Input("a: T")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Op2")
    .Input("a: T")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Op3")
    .Input("c: T")
    .SetShapeFn(shape_inference::UnchangedShape);
)opdef";

  const string op_def_text_with_doc = R"opdef(
REGISTER_OP("Op1")
    .Input("a: T")
    .Doc(R"doc(
Summary for Op1.
)doc")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Op2")
    .Input("a: T")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Summary for Op2.
)doc");

REGISTER_OP("Op3")
    .Input("c: T")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Summary for Op3.
)doc");
)opdef";

  const string op1_text = R"(
name: "Op1"
input_arg {
  name: "a"
}
summary: "Summary for Op1."
)";
  const string op2_text = R"(
name: "Op2"
input_arg {
  name: "a"
}
summary: "Summary for Op2."
)";
  const string op3_text = R"(
name: "Op3"
input_arg {
  name: "c"
}
summary: "Summary for Op3."
)";
  OpDef op1, op2, op3;
  protobuf::TextFormat::ParseFromString(op1_text, &op1);  // NOLINT
  protobuf::TextFormat::ParseFromString(op2_text, &op2);  // NOLINT
  protobuf::TextFormat::ParseFromString(op3_text, &op3);  // NOLINT

  string updated_text =
      RemoveDoc(op2, op_def_text_with_doc,
                op_def_text_with_doc.find("Op2") /* start_location */);
  EXPECT_EQ(string::npos, updated_text.find("Summary for Op2"));
  EXPECT_NE(string::npos, updated_text.find("Summary for Op1"));
  EXPECT_NE(string::npos, updated_text.find("Summary for Op3"));

  updated_text = RemoveDoc(op3, updated_text,
                           updated_text.find("Op3") /* start_location */);
  updated_text = RemoveDoc(op1, updated_text,
                           updated_text.find("Op1") /* start_location */);
  EXPECT_EQ(op_def_text, updated_text);
}

TEST(UpdateApiDefTest, TestCreateApiDef) {
  const string op_text = R"(
name: "Op1"
input_arg {
  name: "a"
  description: "Description for a."
}
output_arg {
  name: "output"
  description: "Description for output."
}
attr {
  name: "b"
  description: "Description for b."
}
summary: "Summary for Op1."
description: "Description\nfor Op1."
)";
  OpDef op;
  protobuf::TextFormat::ParseFromString(op_text, &op);  // NOLINT

  const string expected_api_def = R"(op {
  graph_op_name: "Op1"
  in_arg {
    name: "a"
    description: <<END
Description for a.
END
  }
  out_arg {
    name: "output"
    description: <<END
Description for output.
END
  }
  attr {
    name: "b"
    description: <<END
Description for b.
END
  }
  summary: "Summary for Op1."
  description: <<END
Description
for Op1.
END
}
)";
  EXPECT_EQ(expected_api_def, CreateApiDef(op));
}

}  // namespace
}  // namespace machina
