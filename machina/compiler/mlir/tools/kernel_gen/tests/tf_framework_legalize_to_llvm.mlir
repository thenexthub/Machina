// RUN: kernel-gen-opt %s -tf-kernel-to-toolchain -reconcile-unrealized-casts -split-input-file | FileCheck %s

// CHECK: toolchain.func @_mlir_ciface_tf_alloc
// CHECK-SAME:  (!toolchain.ptr, i64, i64, i32, i32, !toolchain.ptr) -> !toolchain.ptr

// CHECK-LABEL: toolchain.func @alloc(
// CHECK-SAME:    [[TF_CTX:%.*]]: !toolchain.ptr,
// CHECK-SAME:    [[SIZE_0:%.*]]: i64,
// CHECK-SAME:    [[SIZE_2:%.*]]: i64) -> [[DESC_TY:!.*]] {
func.func @alloc(%ctx: !tf_framework.op_kernel_context,
                %size_0 : index , %size_2 : index) -> memref<?x10x?xf32> {
  %buf = tf_framework.alloc(%ctx, %size_0, %size_2) : memref<?x10x?xf32>
  func.return %buf : memref<?x10x?xf32>
}
// Compute number of elements.
// CHECK: [[SIZE_1A:%.*]] = toolchain.mlir.constant(10 : index) : i64
// CHECK: [[SIZE_1B:%.*]] = toolchain.mlir.constant(10 : index) : i64
// CHECK: [[NUM_ELEM_0:%.*]] = toolchain.mul [[SIZE_0]], [[SIZE_1B]] : i64
// CHECK: [[NUM_ELEMS:%.*]] = toolchain.mul [[NUM_ELEM_0]], [[SIZE_2]] : i64

// Compute the size of an individual element.
// CHECK: [[NULL:%.*]] = toolchain.mlir.zero : !toolchain.ptr
// CHECK: [[GEP:%.*]] = toolchain.getelementptr [[NULL]]{{\[}}1]
// CHECK-SAME:            (!toolchain.ptr) -> !toolchain.ptr, f32
// CHECK: [[SIZE_OF_FLOAT:%.*]] = toolchain.ptrtoint [[GEP]]
// CHECK-SAME:            !toolchain.ptr to i64

// Compute output index (-1) and candidate indices (0, NULL).
// CHECK: [[OUTPUT_INDEX:%.*]] = toolchain.mlir.constant(-1 : i32) : i32
// CHECK-NEXT: [[NUM_CANDIDATES:%.*]] = toolchain.mlir.constant(0 : i32) : i32
// CHECK-NEXT: [[CANDIDATES_PTR:%.*]] = toolchain.mlir.zero : !toolchain.ptr

// Allocate memory.
// CHECK: [[BYTES_PTR:%.*]] = toolchain.call @{{.*}}([[TF_CTX]], [[NUM_ELEMS]],
// CHECK-SAME: [[SIZE_OF_FLOAT]], [[OUTPUT_INDEX]], [[NUM_CANDIDATES]],
// CHECK-SAME: [[CANDIDATES_PTR]])

// Build memref descriptor.
// CHECK: [[DESC_0:%.*]] = toolchain.mlir.poison : [[DESC_TY]]

// Set pointers and offset.
// CHECK: [[DESC_1:%.*]] = toolchain.insertvalue [[BYTES_PTR]], [[DESC_0]][0]
// CHECK: [[DESC_2:%.*]] = toolchain.insertvalue [[BYTES_PTR]], [[DESC_1]][1]
// CHECK: [[C0:%.*]] = toolchain.mlir.constant(0 : index) : i64
// CHECK: [[DESC_3:%.*]] = toolchain.insertvalue [[C0]], [[DESC_2]][2] : [[DESC_TY]]

// Set sizes and strides.
// CHECK: [[STRIDE_2:%.*]] = toolchain.mlir.constant(1 : index) : i64
// CHECK: [[DESC_4:%.*]] = toolchain.insertvalue [[SIZE_2]], [[DESC_3]][3, 2]
// CHECK: [[DESC_5:%.*]] = toolchain.insertvalue [[STRIDE_2]], [[DESC_4]][4, 2]
// CHECK: [[STRIDE_1:%.*]] = toolchain.mul [[STRIDE_2]], [[SIZE_2]] : i64
// CHECK: [[DESC_6:%.*]] = toolchain.insertvalue [[SIZE_1A]], [[DESC_5]][3, 1]
// CHECK: [[DESC_7:%.*]] = toolchain.insertvalue [[STRIDE_1]], [[DESC_6]][4, 1]
// CHECK: [[STRIDE_0:%.*]] = toolchain.mul [[STRIDE_1]], [[SIZE_1A]] : i64
// CHECK: [[DESC_8:%.*]] = toolchain.insertvalue [[SIZE_0]], [[DESC_7]][3, 0]
// CHECK: [[DESC_9:%.*]] = toolchain.insertvalue [[STRIDE_0]], [[DESC_8]][4, 0]
// CHECK: toolchain.return [[DESC_9]] : [[DESC_TY]]

// -----

// CHECK: toolchain.func @_mlir_ciface_tf_dealloc(!toolchain.ptr, !toolchain.ptr)

// CHECK-LABEL: toolchain.func @dealloc(
// CHECK-SAME:    [[TF_CTX:%[a-z0-9]*]]: !toolchain.ptr
func.func @dealloc(%ctx: !tf_framework.op_kernel_context,
                  %memref : memref<?x10xf32>) {
  tf_framework.dealloc(%ctx, %memref) : memref<?x10xf32>
  func.return
}
// Extract allocated ptr from the memref descriptor.
// CHECK: %{{.*}} = toolchain.mlir.poison : [[DESC_TY:!.*]]
// CHECK: [[FLOAT_PTR:%.*]] = toolchain.extractvalue %{{.*}}[0] : [[DESC_TY]]

// Deallocate.
// CHECK: toolchain.call @_mlir_ciface_tf_dealloc(
// CHECK-SAME: [[TF_CTX]], [[FLOAT_PTR]]) : (!toolchain.ptr, !toolchain.ptr) -> ()

// -----

// CHECK-LABEL: toolchain.func @_mlir_ciface_tf_report_error(!toolchain.ptr, i32, !toolchain.ptr)
// CHECK: toolchain.mlir.global internal constant [[MSG_CONST:@error_message_[0-9]+]]("Everything is awesome\00")

func.func @report_error(%ctx: !tf_framework.op_kernel_context) {
  tf_framework.report_error %ctx, "INVALID_ARGUMENT", "Everything is awesome" loc(unknown)
  func.return
}
// CHECK:     toolchain.func @report_error([[CTX:%.*]]: !toolchain.ptr)
// CHECK-NEXT:  [[ADDR:%.*]] = toolchain.mlir.addressof [[MSG_CONST]]
// CHECK:       [[MSG:%.*]] = toolchain.getelementptr [[ADDR]]
// CHECK:       [[CODE:%.*]] = toolchain.mlir.constant({{.*}}) : i32
// CHECK:       toolchain.call @{{.*}}_tf_report_error([[CTX]], [[CODE]], [[MSG]])

// -----

// CHECK-LABEL: toolchain.func @unranked_null_memref()
func.func @unranked_null_memref() {
  %null = tf_framework.null_memref : memref<*xf32>
  func.return
}
// CHECK: [[C0:%.*]] = toolchain.mlir.constant(0 : index) : i64
// CHECK: [[DESC_0:%.*]] = toolchain.mlir.poison : !toolchain.struct<(i64, ptr)>
// CHECK: [[DESC_1:%.*]] = toolchain.insertvalue [[C0]], [[DESC_0]][0]
// CHECK: [[PTR:%.*]] = toolchain.alloca {{.*}} x i8
// CHECK: [[DESC_2:%.*]] = toolchain.insertvalue [[PTR]], [[DESC_1]][1]

// -----

// CHECK-LABEL: toolchain.func @ranked_null_memref()
func.func @ranked_null_memref() {
  %null = tf_framework.null_memref : memref<2x?xf32>
  func.return
}
// CHECK: %[[C0:.*]] = toolchain.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[C1:.*]] = toolchain.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[C2:.*]] = toolchain.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[C1_:.*]] = toolchain.mlir.constant(1 : index) : i64

// CHECK: toolchain.mlir.zero
// CHECK: %[[NULL:.*]] = toolchain.mlir.zero : !toolchain.ptr
// CHECK-NEXT: %[[DESC_0:.*]] = toolchain.mlir.poison :
// CHECK-SAME:   !toolchain.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[DESC_1:.*]] = toolchain.insertvalue %[[NULL]], %[[DESC_0]][0]
// CHECK-NEXT: %[[DESC_2:.*]] = toolchain.insertvalue %[[NULL]], %[[DESC_1]][1]
// CHECK-NEXT: %[[DESC_3:.*]] = toolchain.insertvalue %[[C0]], %[[DESC_2]][2]
// CHECK-NEXT: %[[DESC_4:.*]] = toolchain.insertvalue %[[C2]], %[[DESC_3]][3, 0]
// CHECK-NEXT: %[[DESC_5:.*]] = toolchain.insertvalue %[[C1]], %[[DESC_4]][4, 0]
// CHECK-NEXT: %[[DESC_6:.*]] = toolchain.insertvalue %[[C1]], %[[DESC_5]][3, 1]
// CHECK-NEXT: %[[DESC_7:.*]] = toolchain.insertvalue %[[C1_]], %[[DESC_6]][4, 1]

// -----

// CHECK-LABEL: toolchain.func @is_valid_memref
func.func @is_valid_memref(%buf: memref<?xf32>) -> i1 {
  %pred = tf_framework.is_valid_memref(%buf) : memref<?xf32> -> i1
  func.return %pred : i1
}
// CHECK: %[[MEMREF:.*]] = toolchain.insertvalue %{{.*}}, %{{.*}}[4, 0]

// CHECK-NEXT: %[[IS_EMPTY:.*]] = toolchain.mlir.constant(false) : i1
// CHECK-NEXT: %[[C0:.*]] = toolchain.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[SIZE:.*]] = toolchain.extractvalue %[[MEMREF]][3, 0]
// CHECK-NEXT: %[[IS_ZERO:.*]] = toolchain.icmp "eq" %[[SIZE]], %[[C0]] : i64
// CHECK-NEXT: %[[IS_EMPTY_:.*]] =  toolchain.or %[[IS_EMPTY]], %[[IS_ZERO]] : i1

// CHECK-NEXT: %[[PTR_F32:.*]] = toolchain.extractvalue %[[MEMREF]][0]
// CHECK-NEXT: %[[NULL_PTR:.*]] = toolchain.mlir.zero : !toolchain.ptr
// CHECK-NEXT: %[[NOT_NULL:.*]] = toolchain.icmp "ne" %[[PTR_F32]], %[[NULL_PTR]]

// CHECK-NEXT: %[[PRED:.*]] = toolchain.or %[[NOT_NULL]], %[[IS_EMPTY_]]  : i1
// CHECK-NEXT: toolchain.return %[[PRED]]

// -----

// CHECK-LABEL: toolchain.func @_mlir_ciface_tf_jit_compile(!toolchain.ptr, !toolchain.ptr, i64, !toolchain.ptr, i64, !toolchain.ptr, i1, i1, i1) -> !toolchain.ptr
// CHECK: toolchain.mlir.global internal constant @[[CODE:jit_module_code_[0-9]+]]("placeholder\00")

// CHECK: @jit_compile_from_str(%[[CTX:.*]]: !toolchain.ptr)
func.func @jit_compile_from_str(%ctx: !tf_framework.op_kernel_context)
    -> !tf_framework.jit_callable {
  // CHECK: %[[ADDR:.*]] = toolchain.mlir.addressof @[[CODE]]
  // CHECK: %[[CODE_PTR:.*]] = toolchain.getelementptr %[[ADDR]][0, 0]

  // Create stack-allocated array for the tile sizes.
  // CHECK: %[[NUM_TILE_SIZES:.*]] = toolchain.mlir.constant(3 : i64)
  // CHECK: %[[TILE_SIZES:.*]] = toolchain.alloca %[[NUM_TILE_SIZES]] x i64
  // CHECK: %[[C0:.*]] = toolchain.mlir.constant(0 : i64)
  // CHECK: %[[PTR:.*]] = toolchain.getelementptr %[[TILE_SIZES]][%[[C0]]]
  // CHECK: %[[C1:.*]] = toolchain.mlir.constant(1 : i64)
  // CHECK: toolchain.store %[[C1]], %[[PTR]]
  // CHECK: %[[C1:.*]] = toolchain.mlir.constant(1 : i64)
  // CHECK: %[[PTR:.*]] = toolchain.getelementptr %[[TILE_SIZES]][%[[C1]]]
  // CHECK: %[[C2:.*]] = toolchain.mlir.constant(2 : i64)
  // CHECK: toolchain.store %[[C2]], %[[PTR]]
  // CHECK: %[[C2:.*]] = toolchain.mlir.constant(2 : i64)
  // CHECK: %[[PTR:.*]] = toolchain.getelementptr %[[TILE_SIZES]][%[[C2]]]
  // CHECK: %[[C3:.*]] = toolchain.mlir.constant(3 : i64)
  // CHECK: toolchain.store %[[C3]], %[[PTR]]

  // Create stack-allocated array for the unroll factors.
  // CHECK: %[[NUM_UNROLL_FACTORS:.*]] = toolchain.mlir.constant(1 : i64) : i64
  // CHECK: %[[UNROLL_FACTORS:.*]] = toolchain.alloca %[[NUM_UNROLL_FACTORS]] x i64
  // CHECK: %[[C0:.*]] = toolchain.mlir.constant(0 : i64)
  // CHECK: %[[PTR:.*]] = toolchain.getelementptr %[[UNROLL_FACTORS]][%[[C0]]]
  // CHECK: %[[C4:.*]] = toolchain.mlir.constant(4 : i64)
  // CHECK: toolchain.store %[[C4]], %[[PTR]]

  // CHECK-DAG: %[[ENABLE_FTZ:.*]] = toolchain.mlir.constant(false)
  // CHECK-DAG: %[[CPU_CODEGEN:.*]] = toolchain.mlir.constant(false)
  // CHECK: %[[RES:.*]] = toolchain.call @_mlir_ciface_tf_jit_compile
  // CHECK-SAME: %[[CTX]], %[[CODE_PTR]],
  // CHECK-SAME: %[[NUM_TILE_SIZES]], %[[TILE_SIZES]],
  // CHECK-SAME: %[[NUM_UNROLL_FACTORS]], %[[UNROLL_FACTORS]],
  // CHECK-SAME: %[[ENABLE_FTZ]], %[[CPU_CODEGEN]]
  // CHECK: toolchain.return %[[RES]]
  %0 = tf_framework.jit_compile_from_str %ctx, "placeholder" {
      tileSizes = [1, 2, 3], unrollFactors = [4],
      enableFtz = false, index64Bit = false, cpuCodegen = false }
  func.return %0 : !tf_framework.jit_callable
}

// -----

// CHECK-LABEL: toolchain.func @_mlir_ciface_tf_jit_execute(!toolchain.ptr, !toolchain.ptr, !toolchain.ptr, i64, !toolchain.ptr)

// CHECK:      @jit_execute
// CHECK-SAME: (%[[CTX:.*]]: !toolchain.ptr, %[[CALLABLE:.*]]: !toolchain.ptr, %[[RANK:.*]]: i64, %[[ARG_DESCR:.*]]: !toolchain.ptr)
func.func @jit_execute(%ctx: !tf_framework.op_kernel_context,
    %callable : !tf_framework.jit_callable, %arg : memref<*xf32>)
    -> memref<*xf32> {
  // CHECK: %[[T0:.*]] = toolchain.mlir.poison
  // CHECK: %[[T1:.*]] = toolchain.insertvalue %[[RANK]], %[[T0]][0]
  // CHECK: %[[ARG:.*]] = toolchain.insertvalue %[[ARG_DESCR]], %[[T1]][1]
  // CHECK: %[[C1:.*]] = toolchain.mlir.constant(1 : i64)
  // CHECK: %[[RESULT_PTR:.*]] = toolchain.alloca %[[C1]] x !toolchain.struct<(i64, ptr)>

  // Copy argument(s) to stack-allocated buffer.
  // CHECK: %[[NUM_ARGS:.*]] = toolchain.mlir.constant(1 : i64)
  // CHECK: %[[ARGS_PTR:.*]] = toolchain.alloca %[[NUM_ARGS]] x !toolchain.struct<(i64, ptr)>
  // CHECK: %[[C0:.*]] = toolchain.mlir.constant(0 : i64)
  // CHECK: %[[ARGS0_PTR:.*]] = toolchain.getelementptr %[[ARGS_PTR]][%[[C0]]]
  // CHECK: toolchain.store %[[ARG]], %[[ARGS0_PTR]]
  // CHECK: toolchain.call @_mlir_ciface_tf_jit_execute(%[[CTX]], %[[CALLABLE]], %[[RESULT_PTR]], %[[NUM_ARGS]], %[[ARGS_PTR]])
  // CHECK: %[[RESULT:.*]] = toolchain.load %[[RESULT_PTR]]

  // Copy unranked memref descriptor to stack-allocated memory.
  // ...
  // CHECK: %[[STACK_RESULT_DESCR:.*]] = toolchain.alloca %[[RESULT_DESCR_SIZE:[0-9]*]] x i8
  // CHECK: %[[RESULT_DESCR:.*]] = toolchain.extractvalue %[[RESULT]][1]
  // CHECK: "toolchain.intr.memcpy"(%[[STACK_RESULT_DESCR]], %[[RESULT_DESCR]], %[[RESULT_DESCR_SIZE]]) <{isVolatile = false}>
  // CHECK: toolchain.call @free(%[[RESULT_DESCR]])
  // CHECK: %[[T0:.*]] = toolchain.mlir.poison
  // CHECK: %[[RANK:.*]] = toolchain.extractvalue %[[RESULT]][0]
  // CHECK: %[[T1:.*]] = toolchain.insertvalue %[[RANK]], %[[T0]][0]
  // CHECK: %[[RESULT:.*]] = toolchain.insertvalue %[[STACK_RESULT_DESCR]], %[[T1]][1]

  // Copy unranked memref descriptor to heap-allocated memory for return.
  // ...
  // CHECK: %[[HEAP_RESULT_DESCR:.*]] = toolchain.call @malloc(%[[RESULT_DESCR_SIZE:[0-9]*]])
  // CHECK: %[[STACK_RESULT_DESCR:.*]] = toolchain.extractvalue %[[RESULT]][1]
  // CHECK: "toolchain.intr.memcpy"(%[[HEAP_RESULT_DESCR]], %[[STACK_RESULT_DESCR]], %[[RESULT_DESCR_SIZE]]) <{isVolatile = false}>
  // CHECK: %[[T0:.*]] = toolchain.mlir.poison
  // CHECK: %[[RANK:.*]] = toolchain.extractvalue %[[RESULT]][0]
  // CHECK: %[[T1:.*]] = toolchain.insertvalue %[[RANK]], %[[T0]][0]
  // CHECK: %[[RESULT:.*]] = toolchain.insertvalue %[[HEAP_RESULT_DESCR]], %[[T1]][1]
  // CHECK: toolchain.return %[[RESULT]]
  %0 = tf_framework.jit_execute ctx(%ctx) %callable(%arg)
      : memref<*xf32> -> memref<*xf32>
  func.return %0 : memref<*xf32>
}
