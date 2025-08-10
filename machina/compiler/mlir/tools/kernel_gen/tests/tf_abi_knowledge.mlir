// RUN: kernel-gen-opt %s -allow-unregistered-dialect -propagate-tf-abi-knowledge-to-kernels -split-input-file | FileCheck %s --check-prefixes=CHECK,ABI
// RUN: kernel-gen-opt %s -allow-unregistered-dialect -propagate-shape-knowledge-to-kernels -split-input-file | FileCheck %s --check-prefixes=CHECK,SHAPE

// The input is taken from what is actually used in kernel generator lowering
// for unary operations.

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK-LABEL: func @abs
  func.func @abs(%ctx: !tf_framework.op_kernel_context, %arg0: memref<*xf32>, %size: index)
      attributes {tf_entry} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %13 = memref.alloca() : memref<1xindex>
    memref.store %size, %13[%c0] : memref<1xindex>
    %14 = memref.reshape %arg0(%13) : (memref<*xf32>, memref<1xindex>) -> memref<?xf32>
    %15 = memref.dim %14, %c0 : memref<?xf32>
    %16 = tf_framework.alloc(%ctx, %15) : memref<?xf32>
    gpu.launch_func @abs_kernel::@abs_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%14 : memref<?xf32>, %16 : memref<?xf32>)
    func.return
  }

  // CHECK-LABEL: gpu.module @abs_kernel
  gpu.module @abs_kernel {
    // CHECK-LABEL: toolchain.func @abs_kernel
    // ABI-SAME: %[[ARG0:.*]]: !toolchain.ptr, %[[ARG1:.*]]: !toolchain.ptr {toolchain.align = 16 : index},
    // ABI-SAME: %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: !toolchain.ptr, %[[ARG6:.*]]: !toolchain.ptr {toolchain.align = 16 : index, toolchain.noalias},
    // ABI-SAME: %[[ARG7:.*]]: i64, %[[ARG8:.*]]: i64, %[[ARG9:.*]]: i64
    // SHAPE-SAME: %[[ARG0:.*]]: !toolchain.ptr, %[[ARG1:.*]]: !toolchain.ptr, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: !toolchain.ptr, %[[ARG6:.*]]: !toolchain.ptr, %[[ARG7:.*]]: i64, %[[ARG8:.*]]: i64, %[[ARG9:.*]]: i64
    toolchain.func @abs_kernel(%arg0: !toolchain.ptr, %arg1: !toolchain.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !toolchain.ptr, %arg6: !toolchain.ptr, %arg7: i64, %arg8: i64, %arg9: i64) attributes {gpu.kernel} {
      // ABI: %[[ZERO:.*]] = toolchain.mlir.constant(0 : index)
      // ABI: %[[ONE:.*]] = toolchain.mlir.constant(1 : index)
      // CHECK: toolchain.mlir.undef
      %0 = toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[ARG1]]
      // SHAPE-NEXT: toolchain.insertvalue %[[ARG0]]
      %1 = toolchain.insertvalue %arg0, %0[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: toolchain.insertvalue %[[ARG1]]
      %2 = toolchain.insertvalue %arg1, %1[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[ZERO]]
      // SHAPE-NEXT: toolchain.insertvalue %[[ARG2]]
      %3 = toolchain.insertvalue %arg2, %2[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: toolchain.insertvalue %[[ARG3]]
      %4 = toolchain.insertvalue %arg3, %3[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[ONE]]
      // SHAPE-NEXT: toolchain.insertvalue %[[ARG4]]
      %5 = toolchain.insertvalue %arg4, %4[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: toolchain.mlir.undef
      %6 = toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[ARG6]]
      // SHAPE-NEXT: toolchain.insertvalue %[[ARG5]]
      %7 = toolchain.insertvalue %arg5, %6[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: toolchain.insertvalue %[[ARG6]]
      %8 = toolchain.insertvalue %arg6, %7[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[ZERO]]
      // SHAPE-NEXT: toolchain.insertvalue %[[ARG7]]
      %9 = toolchain.insertvalue %arg7, %8[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[ARG8]]
      // SHAPE-NEXT: toolchain.insertvalue %[[ARG3]]
      %10 = toolchain.insertvalue %arg8, %9[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[ONE]]
      // SHAPE-NEXT: toolchain.insertvalue %[[ARG4]]
      %11 = toolchain.insertvalue %arg9, %10[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      toolchain.return
      // CHECK-NEXT: toolchain.return
    }
  }
}

// -----

// Binary op without broadcasting (same shape).

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK-LABEL: func @add_same_shape
  func.func @add_same_shape(%arg0: !tf_framework.op_kernel_context, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %size: index)
      attributes {tf_entry} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %82 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%size], strides: [%c1]: memref<*xf32> to memref<?xf32, strided<[?], offset: 0>>
    %83 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%size], strides: [%c1]: memref<*xf32> to memref<?xf32, strided<[?], offset: 0>>
    %84 = tf_framework.alloc(%arg0, %size) : memref<?xf32>
    gpu.launch_func  @AddV2_kernel_1::@AddV2_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%size : index, %82 : memref<?xf32, strided<[?], offset: 0>>, %83 : memref<?xf32, strided<[?], offset: 0>>, %84 : memref<?xf32>)
    func.return
  }

  // CHECK-LABEL: gpu.module @AddV2_kernel_1
  gpu.module @AddV2_kernel_1 {
    // CHECK-LABEL: toolchain.func @AddV2_kernel
    // ABI-SAME: {toolchain.align = 16 : index}
    // ABI-SAME: {toolchain.align = 16 : index}
    // ABI-SAME: {toolchain.align = 16 : index, toolchain.noalias}
    toolchain.func @AddV2_kernel(%arg0: i64, %arg1: !toolchain.ptr, %arg2: !toolchain.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !toolchain.ptr, %arg7: !toolchain.ptr, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !toolchain.ptr, %arg12: !toolchain.ptr, %arg13: i64, %arg14: i64, %arg15: i64) attributes {gpu.kernel} {
      // ABI: %[[C0:.*]] = toolchain.mlir.constant(0 : index) : i64
      // ABI: %[[C1:.*]] = toolchain.mlir.constant(1 : index) : i64
      %0 = toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %1 = toolchain.insertvalue %arg1, %0[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %2 = toolchain.insertvalue %arg2, %1[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %3 = toolchain.insertvalue %arg3, %2[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %4 = toolchain.insertvalue %arg4, %3[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %5 = toolchain.insertvalue %arg5, %4[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR0:.*]], %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR0]], %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C0]], %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C1]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[SHP:.*]], %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[STR:.*]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %6 = toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %7 = toolchain.insertvalue %arg6, %6[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %8 = toolchain.insertvalue %arg7, %7[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %9 = toolchain.insertvalue %arg8, %8[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %10 = toolchain.insertvalue %arg9, %9[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %11 = toolchain.insertvalue %arg10, %10[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR1:.*]], %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR1]], %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C0]], %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C1]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[SHP]], %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NOT: toolchain.insertvalue %[[STR]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %12 = toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %13 = toolchain.insertvalue %arg11, %12[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %14 = toolchain.insertvalue %arg12, %13[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %15 = toolchain.insertvalue %arg13, %14[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %16 = toolchain.insertvalue %arg14, %15[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %17 = toolchain.insertvalue %arg15, %16[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR2:.*]], %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR2]], %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C0]], %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C1]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[SHP]], %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[STR1:.*]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NOT: toolchain.insertvalue %[[STR]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      toolchain.return
      // CHECK-NEXT: toolchain.return
    }
  }
}

// -----

// Test binary op with broadcasting - 2d case.

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK-LABEL: func @add_same_shape
  func.func @add_same_shape(%arg0: !tf_framework.op_kernel_context, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %size0: index, %size1: index, %stride0: index, %stride1: index)
      attributes {tf_entry} {
    %c1 = arith.constant 1 : index
    %216 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%size1, %size0], strides: [%size0, %c1]: memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: 0>>
    %241 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%size1, %size0], strides: [%size0, %c1]: memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: 0>>
    %304 = memref.reinterpret_cast %216 to offset: [0], sizes: [%size1, %size0], strides: [%stride1, %stride0]: memref<?x?xf32, strided<[?, ?], offset: 0>> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>>
    %309 = memref.reinterpret_cast %241 to offset: [0], sizes: [%size1, %size0], strides: [%stride0, %stride1]: memref<?x?xf32, strided<[?, ?], offset: 0>> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>>
    %310 = tf_framework.alloc(%arg0, %size1, %size0) : memref<?x?xf32>
    gpu.launch_func  @AddV2_kernel_3::@AddV2_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%size0 : index, %size1 : index, %310 : memref<?x?xf32>, %304 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>>, %309 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>>)
    func.return
  }

  // CHECK-LABEL: gpu.module @AddV2_kernel_3
  gpu.module @AddV2_kernel_3 {
    // CHECK-LABEL: toolchain.func @AddV2_kernel
    // ABI-SAME: {toolchain.align = 16 : index, toolchain.noalias}
    // ABI-SAME: {toolchain.align = 16 : index}
    // ABI-SAME: {toolchain.align = 16 : index}
    toolchain.func @AddV2_kernel(%arg0: i64, %arg1: i64, %arg2: !toolchain.ptr, %arg3: !toolchain.ptr {toolchain.align = 16 : index, toolchain.noalias}, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !toolchain.ptr, %arg10: !toolchain.ptr {toolchain.align = 16 : index}, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !toolchain.ptr, %arg17: !toolchain.ptr {toolchain.align = 16 : index}, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel} {
      // ABI: %[[C0:.*]] = toolchain.mlir.constant(0 : index) : i64
      // ABI: %[[C1:.*]] = toolchain.mlir.constant(1 : index) : i64
      %0 = toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %1 = toolchain.insertvalue %arg2, %0[0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %2 = toolchain.insertvalue %arg3, %1[1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %3 = toolchain.insertvalue %arg4, %2[2] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %4 = toolchain.insertvalue %arg5, %3[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %5 = toolchain.insertvalue %arg7, %4[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %6 = toolchain.insertvalue %arg6, %5[3, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %7 = toolchain.insertvalue %arg8, %6[4, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR0:.*]], %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR0]], %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C0]], %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[3, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C1]], %{{.*}}[4, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[SHP0:.*]], %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[STR0:.*]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[SHP1:.*]], %{{.*}}[3, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[STR1:.*]], %{{.*}}[4, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %8 = toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %9 = toolchain.insertvalue %arg9, %8[0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %10 = toolchain.insertvalue %arg10, %9[1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %11 = toolchain.insertvalue %arg11, %10[2] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %12 = toolchain.insertvalue %arg12, %11[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %13 = toolchain.insertvalue %arg14, %12[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %14 = toolchain.insertvalue %arg13, %13[3, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %15 = toolchain.insertvalue %arg15, %14[4, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR0:.*]], %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR0]], %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C0]], %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NOT: toolchain.insertvalue %[[C1]], %{{.*}}[3, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[SHP0]], %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NOT: toolchain.insertvalue %[[STR0]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE: toolchain.insertvalue %[[SHP1]], %{{.*}}[3, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NOT: toolchain.insertvalue %[[STR1]], %{{.*}}[4, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %16 = toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %17 = toolchain.insertvalue %arg16, %16[0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %18 = toolchain.insertvalue %arg17, %17[1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %19 = toolchain.insertvalue %arg18, %18[2] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %20 = toolchain.insertvalue %arg19, %19[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %21 = toolchain.insertvalue %arg21, %20[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %22 = toolchain.insertvalue %arg20, %21[3, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %23 = toolchain.insertvalue %arg22, %22[4, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR0:.*]], %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR0]], %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C0]], %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[3, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NOT: toolchain.insertvalue %[[C1]], %{{.*}}[4, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[SHP0]], %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NOT: toolchain.insertvalue %[[STR0]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE: toolchain.insertvalue %[[SHP1]], %{{.*}}[3, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NOT: toolchain.insertvalue %[[STR1]], %{{.*}}[4, 1] : !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      toolchain.return
      // CHECK: toolchain.return
    }
  }
}

// -----


// Test binary op with broadcasting a scalar.

#map0 = affine_map<(d0)[s0] -> (d0 * s0)>

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK-LABEL: func @add_one_scalar
  func.func @add_one_scalar(%arg0: !tf_framework.op_kernel_context, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %size: index)
      attributes {tf_entry} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %13 = memref.cast %arg1 : memref<*xf32> to memref<f32>
    %26 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%size], strides: [%c1]: memref<*xf32> to memref<?xf32, strided<[?], offset: 0>>
    %27 = memref.reinterpret_cast %13 to offset: [0], sizes: [%size], strides: [%c0]: memref<f32> to memref<?xf32, #map0>
    %28 = memref.reinterpret_cast %26 to offset: [0], sizes: [%size], strides: [%c1]: memref<?xf32, strided<[?], offset: 0>> to memref<?xf32, #map0>
    %29 = tf_framework.alloc(%arg0, %size) : memref<?xf32>
    gpu.launch_func  @AddV2_kernel::@AddV2_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%size : index, %29 : memref<?xf32>, %27 : memref<?xf32, #map0>, %28 : memref<?xf32, #map0>)
    func.return
  }
  // CHECK-LABEL: gpu.module @AddV2_kernel
  gpu.module @AddV2_kernel {
    // CHECK-LABEL: toolchain.func @AddV2_kernel
    // ABI-SAME: {toolchain.align = 16 : index, toolchain.noalias}
    // ABI-SAME: {toolchain.align = 16 : index}
    // ABI-SAME: {toolchain.align = 16 : index}
    toolchain.func @AddV2_kernel(%arg0: i64, %arg1: !toolchain.ptr, %arg2: !toolchain.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !toolchain.ptr, %arg7: !toolchain.ptr, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !toolchain.ptr, %arg12: !toolchain.ptr, %arg13: i64, %arg14: i64, %arg15: i64) attributes {gpu.kernel} {
      // ABI: %[[C0:.*]] = toolchain.mlir.constant(0 : index) : i64
      // ABI: %[[C1:.*]] = toolchain.mlir.constant(1 : index) : i64
      %0 = toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %1 = toolchain.insertvalue %arg1, %0[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %2 = toolchain.insertvalue %arg2, %1[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %3 = toolchain.insertvalue %arg3, %2[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %4 = toolchain.insertvalue %arg4, %3[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %5 = toolchain.insertvalue %arg5, %4[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR0:.*]], %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR0]], %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C0]], %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C1]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[SHP:.*]], %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[STR:.*]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %6 = toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %7 = toolchain.insertvalue %arg6, %6[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %8 = toolchain.insertvalue %arg7, %7[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %9 = toolchain.insertvalue %arg8, %8[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %10 = toolchain.insertvalue %arg9, %9[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %11 = toolchain.insertvalue %arg10, %10[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR1:.*]], %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR1]], %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C0]], %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C0]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[SHP]], %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NOT: toolchain.insertvalue %[[STR]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %12 = toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %13 = toolchain.insertvalue %arg11, %12[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %14 = toolchain.insertvalue %arg12, %13[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %15 = toolchain.insertvalue %arg13, %14[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %16 = toolchain.insertvalue %arg14, %15[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %17 = toolchain.insertvalue %arg15, %16[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR2:.*]], %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[PTR2]], %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C0]], %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: toolchain.insertvalue %[[C1]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: toolchain.mlir.undef : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[1] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %{{.*}}, %{{.*}}[2] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: toolchain.insertvalue %[[SHP]], %{{.*}}[3, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NOT: toolchain.insertvalue %[[STR]], %{{.*}}[4, 0] : !toolchain.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      toolchain.return
      // CHECK: toolchain.return
    }
  }
}

