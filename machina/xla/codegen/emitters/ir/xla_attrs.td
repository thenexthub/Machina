/* Copyright 2024 The OpenXLA Authors.

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

#ifndef MACHINA_MACHINA_XLA_CODEGEN_EMITTERS_IR_MACHINA_MACHINA_XLA_ATTRS
#define MACHINA_MACHINA_XLA_CODEGEN_EMITTERS_IR_MACHINA_MACHINA_XLA_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "machina/xla/codegen/emitters/ir/xla_dialect.td"

class MACHINA_MACHINA_XLA_Attr<string name, list<Trait> traits = []> :
      AttrDef<XlaDialect, name, traits> {
}

def MACHINA_MACHINA_XLA_AffineMapParameter :
    AttrOrTypeParameter<"::mlir::AffineMap", ""> {
}

def MACHINA_MACHINA_XLA_IndexingMapVariableParameter
  : ArrayRefParameter<"::xla::IndexingMap::Variable",
                      "IndexingMapVariableArray"> {
}

def MACHINA_MACHINA_XLA_ConstraintsParameter :
    ArrayRefParameter<"::std::pair<::mlir::AffineExpr, ::xla::Interval>",
                      "ContraintsArray"> {
}

def MACHINA_MACHINA_XLA_IndexingMapAttr : MACHINA_MACHINA_XLA_Attr<"IndexingMap"> {
  let summary = "An Attribute representing an indexing map.";
  let mnemonic = "indexing_map";
  let description = [{This attribute stores an indexing map. See
    https://openxla.org/xla/indexing for more details.
  }];
  let parameters = (ins MACHINA_MACHINA_XLA_AffineMapParameter:$map,
                        MACHINA_MACHINA_XLA_IndexingMapVariableParameter:$dim_vars,
                        MACHINA_MACHINA_XLA_IndexingMapVariableParameter:$range_vars,
                        MACHINA_MACHINA_XLA_ConstraintsParameter:$constraints);
  let hasCustomAssemblyFormat = 1;
  let builders =  [
    AttrBuilder<(ins "const ::xla::IndexingMap&":$indexing_map)>,
  ];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    // Returns the indexing map constructed from IndexingMapAttr.
    xla::IndexingMap getIndexingMap() const;

    // Returns the number of indexing map results.
    int64_t getNumResults() const;
  }];
}

// Note: settle for BackendKind to avoid clashing with the existing
// xla::Backend and xla::BackendType types.
def MACHINA_MACHINA_XLA_BackendKind : I32EnumAttr<"BackendKind", "XLA Backend kind (or type)", [
    I32EnumAttrCase<"kCpu", 0, "cpu">,
    I32EnumAttrCase<"kGpu", 1, "gpu">,
    I32EnumAttrCase<"kTpu", 2, "tpu">
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::xla";
}
def MACHINA_MACHINA_XLA_BackendKindAttr :
    EnumAttr<XlaDialect, MACHINA_MACHINA_XLA_BackendKind, "backend_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def MACHINA_MACHINA_XLA_ExtraBackendOptionsAttr
  : ArrayOfAttr<XlaDialect,
                "ExtraBackendOptions",
                "extra_backend_options",
                "mlir::StringAttr"> {
}

def MACHINA_MACHINA_XLA_CpuMemoryRegionNameAttr
  : AttrDef<XlaDialect, "CpuMemoryRegionName", [], "mlir::StringAttr"> {
  let mnemonic = "cpu_memory_region_name";
}

def WorkGroupDimension : I32EnumAttr<"WorkGroupDimension",
    "a dimension, either 'x', 'y', or 'z'",
    [
      I32EnumAttrCase<"x", 0>,
      I32EnumAttrCase<"y", 1>,
      I32EnumAttrCase<"z", 2>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::xla";
}
def WorkGroupDimensionAttr : EnumAttr<XlaDialect, WorkGroupDimension, "dim">;

#endif // MACHINA_MACHINA_XLA_CODEGEN_EMITTERS_IR_MACHINA_MACHINA_XLA_ATTRS
