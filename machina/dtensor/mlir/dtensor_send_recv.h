/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_DTENSOR_MLIR_DTENSOR_SEND_RECV_H_
#define MACHINA_DTENSOR_MLIR_DTENSOR_SEND_RECV_H_

#include "absl/status/status.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"

namespace machina {
namespace dtensor {

// Given DTensorSend or DTensorRecv op, returns the corresponding DTensorRecv
// or DTensorSend op with the same key.
template <typename DTensorOp>
StatusOr<mlir::Operation*> GetCorrespondingDTensorSendRecvOp(
    mlir::ModuleOp module, DTensorOp dtensor_op) {
  mlir::Operation* corresponding_op = nullptr;
  if (std::is_same<DTensorOp, mlir::TF::DTensorSend>::value) {
    module.walk([&](mlir::Operation* op) {
      if (auto xla_recv_tpu = toolchain::dyn_cast<mlir::TF::XlaRecvFromHostOp>(op)) {
        if (dtensor_op.getKey() == xla_recv_tpu.getKey()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto xla_recv_cpu =
                     toolchain::dyn_cast<mlir::TF::_XlaRecvAtHostV2Op>(op)) {
        if (dtensor_op.getKey() == xla_recv_cpu.getKey()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto dtensor_recv =
                     toolchain::dyn_cast<mlir::TF::DTensorRecv>(op)) {
        if (dtensor_op.getKey() == dtensor_recv.getKey()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto host_recv = toolchain::dyn_cast<mlir::TF::_HostRecvOp>(op)) {
        if (dtensor_op.getKey() == host_recv.getTensorName()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      }
      return mlir::WalkResult::advance();
    });
  } else {
    const bool is_recv = std::is_same<DTensorOp, mlir::TF::DTensorRecv>::value;
    if (!is_recv) {
      return absl::InternalError(
          "Error checking if is same for DTensorOp and DTensorRecv.");
    }
    module.walk([&](mlir::Operation* op) {
      if (auto xla_send_tpu = toolchain::dyn_cast<mlir::TF::XlaSendToHostOp>(op)) {
        if (dtensor_op.getKey() == xla_send_tpu.getKey()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto xla_send_cpu =
                     toolchain::dyn_cast<mlir::TF::_XlaSendFromHostV2Op>(op)) {
        if (dtensor_op.getKey() == xla_send_cpu.getKey()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto dtensor_send =
                     toolchain::dyn_cast<mlir::TF::DTensorSend>(op)) {
        if (dtensor_op.getKey() == dtensor_send.getKey()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto host_send = toolchain::dyn_cast<mlir::TF::_HostSendOp>(op)) {
        if (dtensor_op.getKey() == host_send.getTensorName()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      }
      return mlir::WalkResult::advance();
    });
  }

  if (!corresponding_op)
    return absl::InvalidArgumentError(
        "DTensorSend/DTensorRecv op must have corresponding "
        "DTensorRecv/DTensorSend op.");

  return corresponding_op;
}

// Lowers DTensorSend to a number of different device-specific ops:
// _HostSend, XlaSendFromHost, XlaSendToHost, etc.
StatusOr<mlir::Operation*> LowerDTensorRecv(mlir::Operation* send_op,
                                            mlir::Operation* recv_op);

// Lowers DTensorRecv to a number of different device-specific ops:
// _HostRecv, XlaRecvAtHost, XlaRecvFromHost, etc.
StatusOr<mlir::Operation*> LowerDTensorSend(mlir::Operation* send_op,
                                            mlir::Operation* recv_op);

// Lowers a DTensorSend and DTensorRecv pair to XLA ops
StatusOr<mlir::Operation*> LowerDTensorSendAndRecv(mlir::Operation* send_op,
                                                   mlir::Operation* recv_op);

}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_MLIR_DTENSOR_SEND_RECV_H_
