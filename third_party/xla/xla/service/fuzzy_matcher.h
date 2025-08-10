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

#ifndef MACHINA_MACHINA_XLA_SERVICE_FUZZY_MATCHER_H_
#define MACHINA_MACHINA_XLA_SERVICE_FUZZY_MATCHER_H_

#include "machina/xla/hlo/ir/hlo_instruction.h"
#include "machina/xla/hlo/ir/hlo_opcode.h"
#include "machina/xla/service/pattern_matcher.h"

namespace xla {

// Fuzzy matchers for HLOs.
namespace fm {

// TODO(b/355972677): Extend this to support opcodes other than convert
template <typename Pattern>
auto OptConvert(Pattern pattern) {
  auto shared = match::SharedSubpattern(pattern);
  return match::AnyOf<HloInstruction>(match::Convert(shared), shared);
}

#define MACHINA_MACHINA_XLA_FUZZY_UNOP_PATTERN(NAME)                                           \
  template <typename HloInstructionType>                                       \
  inline auto NAME(HloInstructionType** matched_inst) {                        \
    return OptConvert(match::Op(matched_inst).WithOpcode(HloOpcode::k##NAME)); \
  }                                                                            \
                                                                               \
  template <typename Arg>                                                      \
  inline auto NAME(Arg&& arg) {                                                \
    return OptConvert(match::Op()                                              \
                          .WithOpcode(HloOpcode::k##NAME)                      \
                          .WithOperand(0, std::forward<Arg>(arg)));            \
  }                                                                            \
                                                                               \
  template <typename HloInstructionType, typename Arg>                         \
  inline auto NAME(HloInstructionType** matched_inst, Arg&& arg) {             \
    return OptConvert(match::Op(matched_inst)                                  \
                          .WithOpcode(HloOpcode::k##NAME)                      \
                          .WithOperand(0, std::forward<Arg>(arg)));            \
  }
MACHINA_MACHINA_XLA_FUZZY_UNOP_PATTERN(Tanh)
MACHINA_MACHINA_XLA_FUZZY_UNOP_PATTERN(Exp)
MACHINA_MACHINA_XLA_FUZZY_UNOP_PATTERN(Broadcast)
#undef MACHINA_MACHINA_XLA_FUZZY_UNOP_PATTERN

#define MACHINA_MACHINA_XLA_FUZZY_BINOP_PATTERN(NAME)                                         \
  template <typename HloInstructionType, typename Lhs, typename Rhs>          \
  inline auto NAME(HloInstructionType** matched_inst, Lhs&& lhs, Rhs&& rhs) { \
    return OptConvert(match::Op(matched_inst)                                 \
                          .WithOpcode(HloOpcode::k##NAME)                     \
                          .WithOperand(0, std::forward<Lhs>(lhs))             \
                          .WithOperand(1, std::forward<Rhs>(rhs)));           \
  }                                                                           \
  template <typename Lhs, typename Rhs>                                       \
  inline auto NAME(Lhs&& lhs, Rhs&& rhs) {                                    \
    return OptConvert(match::Op()                                             \
                          .WithOpcode(HloOpcode::k##NAME)                     \
                          .WithOperand(0, std::forward<Lhs>(lhs))             \
                          .WithOperand(1, std::forward<Rhs>(rhs)));           \
  }
MACHINA_MACHINA_XLA_FUZZY_BINOP_PATTERN(Dot)
MACHINA_MACHINA_XLA_FUZZY_BINOP_PATTERN(Divide)
MACHINA_MACHINA_XLA_FUZZY_BINOP_PATTERN(Subtract)
MACHINA_MACHINA_XLA_FUZZY_BINOP_PATTERN(Multiply)
// Currently we only use binary matcher for reduce.
MACHINA_MACHINA_XLA_FUZZY_BINOP_PATTERN(Reduce)
#undef MACHINA_MACHINA_XLA_FUZZY_BINOP_PATTERN

#define MACHINA_MACHINA_XLA_FUZZY_TERNOP_PATTERN(NAME)                                 \
  template <typename Arg0, typename Arg1, typename Arg2>               \
  inline auto NAME(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2) {            \
    return OptConvert(match::Op()                                      \
                          .WithOpcode(HloOpcode::k##NAME)              \
                          .WithOperand(0, std::forward<Arg0>(arg0))    \
                          .WithOperand(1, std::forward<Arg1>(arg1))    \
                          .WithOperand(2, std::forward<Arg2>(arg2)));  \
  }                                                                    \
                                                                       \
  template <typename HloInstructionType, typename Arg0, typename Arg1, \
            typename Arg2>                                             \
  inline auto NAME(HloInstructionType** matched_inst, Arg0&& arg0,     \
                   Arg1&& arg1, Arg2&& arg2) {                         \
    return OptConvert(match::Op(matched_inst)                          \
                          .WithOpcode(HloOpcode::k##NAME)              \
                          .WithOperand(0, std::forward<Arg0>(arg0))    \
                          .WithOperand(1, std::forward<Arg1>(arg1))    \
                          .WithOperand(2, std::forward<Arg2>(arg2)));  \
  }
MACHINA_MACHINA_XLA_FUZZY_TERNOP_PATTERN(Select);
#undef MACHINA_MACHINA_XLA_FUZZY_TERNOP_PATTERN

}  // namespace fm

}  // namespace xla

#endif  // MACHINA_MACHINA_XLA_SERVICE_FUZZY_MATCHER_H_
