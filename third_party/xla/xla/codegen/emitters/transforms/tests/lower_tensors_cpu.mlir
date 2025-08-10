// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="target_type=cpu" \
// RUN: | FileCheck %s

func.func @load_non_gep_from_args(%arg0: !toolchain.ptr) -> !toolchain.ptr {
  %0 = toolchain.getelementptr inbounds %arg0[1]
    : (!toolchain.ptr) -> !toolchain.ptr, !toolchain.ptr
  %1 = toolchain.load %0 : !toolchain.ptr -> !toolchain.ptr
  %2 = toolchain.load %1 : !toolchain.ptr -> !toolchain.ptr
  func.return %2 : !toolchain.ptr
}

// CHECK-LABEL: @load_non_gep_from_args
// CHECK-NEXT:    %0 = toolchain.getelementptr inbounds %arg0[1]
// CHECK-NEXT:    %1 = toolchain.load %0 : !toolchain.ptr -> !toolchain.ptr
// CHECK-NEXT:    %2 = toolchain.load %1 : !toolchain.ptr -> !toolchain.ptr
// CHECK-NEXT:    return %2 : !toolchain.ptr
