/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include "machina/c/experimental/saved_model/core/tf_concrete_function_test_protos.h"

#include <string>

#include "absl/strings/string_view.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/protobuf/struct.pb.h"

namespace machina {
namespace testing {
namespace {

constexpr absl::string_view kZeroArgInputSignatureTextProto = R"(
tuple_value: {
    values: {
      tuple_value: {
      }
    }
    values: {
      dict_value: {
      }
    }
}
)";

constexpr absl::string_view kSingleArgInputSignatureTextProto = R"(
tuple_value: {
    values: {
    tuple_value: {
        values: {
        tensor_spec_value: {
            name : "x"
            shape: {
            dim: {
                size: 1
            }
            dim: {
                size: 10
            }
            }
            dtype: DT_FLOAT
        }
        }
    }
    }
    values: {
    dict_value: {
    }
    }
}
)";

constexpr absl::string_view kThreeArgInputSignatureTextProto = R"(
tuple_value: {
  values: {
    tuple_value: {
      values: {
        tensor_spec_value: {
          name : "x"
          shape: {
            dim: {
              size: 1
            }
          }
          dtype: DT_FLOAT
        }
      }
      values: {
        tensor_spec_value: {
          name : "y"
          shape: {
            dim: {
              size: 1
            }
          }
          dtype: DT_FLOAT
        }
      }
      values: {
        tensor_spec_value: {
          name : "z"
          shape: {
            dim: {
              size: 1
            }
          }
          dtype: DT_FLOAT
        }
      }
    }
  }
  values: {
    dict_value: {
    }
  }
}

)";

constexpr absl::string_view kZeroReturnOutputSignatureTextProto = R"(
none_value: {}
)";

constexpr absl::string_view kSingleReturnOutputSignatureTextProto = R"(
tensor_spec_value: {
  shape: {
    dim: {
        size: 1
    }
  }
  dtype: DT_FLOAT
}
)";

constexpr absl::string_view kThreeReturnOutputSignatureTextProto = R"(
tuple_value: {
    values: {
    dict_value: {
        fields: {
        key  : "a"
        value: {
            tensor_spec_value: {
            name : "0/a"
            shape: {
                dim: {
                size: 1
                }
            }
            dtype: DT_FLOAT
            }
        }
        }
        fields: {
        key  : "b"
        value: {
            tensor_spec_value: {
            name : "0/b"
            shape: {
                dim: {
                size: 1
                }
            }
            dtype: DT_FLOAT
            }
        }
        }
    }
    }
    values: {
    tensor_spec_value: {
        name : "1"
        shape: {
        dim: {
            size: 1
        }
        }
        dtype: DT_FLOAT
    }
    }
}
)";

StructuredValue ParseStructuredValue(absl::string_view text_proto) {
  StructuredValue value;
  CHECK(machina::protobuf::TextFormat::ParseFromString(string(text_proto),
                                                          &value));
  return value;
}

}  // namespace

StructuredValue ZeroArgInputSignature() {
  return ParseStructuredValue(kZeroArgInputSignatureTextProto);
}

StructuredValue SingleArgInputSignature() {
  return ParseStructuredValue(kSingleArgInputSignatureTextProto);
}

StructuredValue ThreeArgInputSignature() {
  return ParseStructuredValue(kThreeArgInputSignatureTextProto);
}

StructuredValue ZeroReturnOutputSignature() {
  return ParseStructuredValue(kZeroReturnOutputSignatureTextProto);
}

StructuredValue SingleReturnOutputSignature() {
  return ParseStructuredValue(kSingleReturnOutputSignatureTextProto);
}

StructuredValue ThreeReturnOutputSignature() {
  return ParseStructuredValue(kThreeReturnOutputSignatureTextProto);
}

}  // namespace testing
}  // namespace machina
