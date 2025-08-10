// RUN: xla-translate-opt %s -xla-legalize-xla-framework-to-toolchain | FileCheck %s

memref.global "private" constant @__constant_xf32 : memref<f32> = dense<42.0>

func.func @buffer_type(%arg: !xla_framework.buffer {xla_framework.input_mapping = 0 : i64})
                      attributes {xla_entry} {
  %val = xla_framework.buffer_to_mem %arg : memref<f32>
  %global = memref.get_global @__constant_xf32 : memref<f32>
  memref.copy %global, %val : memref<f32> to memref<f32>
  func.return
}

// CHECK-LABEL: @buffer_type
// The following signature is always the same.
// CHECK-SAME: %{{[^:]*}}: !toolchain.ptr,
// CHECK-SAME: %{{[^:]*}}: !toolchain.ptr,
// CHECK-SAME: %{{[^:]*}}: !toolchain.ptr,
// CHECK-SAME: %[[BUFFERS:[^:]*]]: !toolchain.ptr,
// CHECK-SAME: %{{[^:]*}}: !toolchain.ptr,
// CHECK-SAME: %{{[^:]*}}: !toolchain.ptr) {
// Retrieve pointer from the input as part of the function signature lowering.
// CHECK: %[[C0:.*]] = toolchain.mlir.constant(0 : i64) : i32
// CHECK: %[[PTRS:.*]] = toolchain.getelementptr %[[BUFFERS]][%[[C0]]] : (!toolchain.ptr, i32) -> !toolchain.ptr
// CHECK: %[[PTR0:.*]] = toolchain.load %[[PTRS]] : !toolchain.ptr
// Create memref descriptor as the buffer_to_mem lowering.
// CHECK: %[[MEMREF:.*]] = toolchain.mlir.poison : !toolchain.struct<(ptr, ptr, i64)>
// CHECK: %[[MEMREF1:.*]] = toolchain.insertvalue %[[PTR0]], %[[MEMREF]][0] : !toolchain.struct<(ptr, ptr, i64)>
// CHECK: %[[MEMREF:.*]] = toolchain.insertvalue %[[PTR0]], %[[MEMREF1]][1] : !toolchain.struct<(ptr, ptr, i64)>
// CHECK: %[[C0_0:.*]] = toolchain.mlir.constant(0 : index) : i64
// CHECK: toolchain.insertvalue %[[C0_0:.*]], %[[MEMREF:.*]][2] : !toolchain.struct<(ptr, ptr, i64)>
// No return values in this case
// CHECK: return


func.func @return_tuple(%result0: !xla_framework.buffer, %result1: !xla_framework.buffer)
                      attributes {xla_entry, xla_framework.result_inner_mapping=[1,2], xla_framework.result_mapping=0} {
  func.return
}


// CHECK-LABEL: @return_tuple
// The following signature is always the same.
// CHECK-SAME: %{{[^:]*}}: !toolchain.ptr,
// CHECK-SAME: %{{[^:]*}}: !toolchain.ptr,
// CHECK-SAME: %{{[^:]*}}: !toolchain.ptr,
// CHECK-SAME: %[[BUFFERS:[^:]*]]: !toolchain.ptr,
// CHECK-SAME: %{{[^:]*}}: !toolchain.ptr,
// CHECK-SAME: %{{[^:]*}}: !toolchain.ptr) {
// Get Tuple
// CHECK-NEXT: %[[C0:.*]] = toolchain.mlir.constant(0 : i64) : i32
// CHECK-NEXT: %[[PTRS0:.*]] = toolchain.getelementptr %[[BUFFERS]][%[[C0]]] : (!toolchain.ptr, i32) -> !toolchain.ptr
// CHECK-NEXT: %[[PTR0:.*]] = toolchain.load %[[PTRS0]] : !toolchain.ptr
// Get individual output buffer
// CHECK-NEXT: %[[C1:.*]] = toolchain.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[PTRS1:.*]] = toolchain.getelementptr %[[BUFFERS]][%[[C1]]] : (!toolchain.ptr, i32) -> !toolchain.ptr
// CHECK-NEXT: %[[PTR1:.*]] = toolchain.load %[[PTRS1]] : !toolchain.ptr
// Store into tuple
// CHECK-NEXT: %[[C0:.*]] = toolchain.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[TUPLE_ELEMENT:.*]] = toolchain.getelementptr %[[PTR0]][%[[C0]]] : (!toolchain.ptr, i32) -> !toolchain.ptr
// CHECK-NEXT: toolchain.store %[[PTR1]], %[[TUPLE_ELEMENT]] : !toolchain.ptr
// Get tuple
// CHECK-NEXT: %[[C0:.*]] = toolchain.mlir.constant(0 : i64) : i32
// CHECK-NEXT: %[[PTRS0:.*]] = toolchain.getelementptr %[[BUFFERS]][%[[C0]]] : (!toolchain.ptr, i32) -> !toolchain.ptr
// CHECK-NEXT: %[[PTR0:.*]] = toolchain.load %[[PTRS0]] : !toolchain.ptr
// Get individual output buffer
// CHECK-NEXT: %[[C2:.*]] = toolchain.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[PTRS2:.*]] = toolchain.getelementptr %[[BUFFERS]][%[[C2]]] : (!toolchain.ptr, i32) -> !toolchain.ptr
// CHECK-NEXT: %[[PTR2:.*]] = toolchain.load %[[PTRS2]] : !toolchain.ptr
// Store into Tuple
// CHECK-NEXT: %[[C1:.*]] = toolchain.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[TUPLE_ELEMENT:.*]] = toolchain.getelementptr %[[PTR0]][%[[C1]]] : (!toolchain.ptr, i32) -> !toolchain.ptr
// CHECK-NEXT: toolchain.store %[[PTR2]], %[[TUPLE_ELEMENT]] : !toolchain.ptr
// No return values
// CHECK-NEXT:  return
