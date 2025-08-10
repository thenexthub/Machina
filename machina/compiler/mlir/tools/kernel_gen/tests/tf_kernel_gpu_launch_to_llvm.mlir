// RUN: kernel-gen-opt %s -tf-kernel-to-toolchain -reconcile-unrealized-casts -split-input-file | FileCheck %s --dump-input=always

// CHECK-LABEL: module @main
module @main attributes {gpu.container_module} {

// CHECK-NOT: gpu.module @kernel_module
gpu.module @kernel_module attributes {gpu.binary_blob = "BLOB!"} {
  toolchain.func @the_kernel() attributes {gpu.kernel} {
    toolchain.return
  }
}

// CHECK: toolchain.func @_mlir_ciface_tf_launch_kernel(!toolchain.ptr, !toolchain.ptr, !toolchain.ptr, i64, i64, i64, i64, i64, i64, !toolchain.ptr)
// CHECK-DAG: toolchain.mlir.global internal constant @kernel_module_the_kernel_kernel_name("the_kernel\00")
// CHECK-DAG: toolchain.mlir.global internal constant @kernel_module_blob("BLOB!")

// CHECK-LABEL: toolchain.func @launch
// CHECK-SAME: (%[[CTX:.*]]: !toolchain.ptr, %{{.*}}: !toolchain.ptr, %{{.*}}: !toolchain.ptr, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64
func.func @launch(%ctx: !tf_framework.op_kernel_context, %memref: memref<?x10xf32>) {
  // CHECK: %[[C1:.*]] = toolchain.mlir.constant(1 : index) : i64
  // CHECK: %[[BLOB:.*]] = toolchain.mlir.addressof @kernel_module_blob : !toolchain.ptr
  // CHECK: %[[BLOB_PTR:.*]] = toolchain.getelementptr %[[BLOB]][0, 0] : (!toolchain.ptr) -> !toolchain.ptr, !toolchain.array<5 x i8>
  // CHECK: %[[NAME:.*]] = toolchain.mlir.addressof @kernel_module_the_kernel_kernel_name : !toolchain.ptr
  // CHECK: %[[NAME_PTR:.*]] = toolchain.getelementptr %[[NAME]][0, 0] : (!toolchain.ptr) -> !toolchain.ptr, !toolchain.array<11 x i8>
  // CHECK: %[[C7:.*]] = toolchain.mlir.constant(7 : i32) : i32
  // CHECK: %[[ARGS:.*]] = toolchain.alloca %22 x !toolchain.ptr : (i32) -> !toolchain.ptr
  // CHECK: toolchain.call @_mlir_ciface_tf_launch_kernel(%[[CTX]], %[[BLOB_PTR]], %[[NAME_PTR]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[ARGS]])
  %c1 = arith.constant 1 : index
  gpu.launch_func  @kernel_module::@the_kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1)
      args(%memref: memref<?x10xf32>)
  func.return
}

}
