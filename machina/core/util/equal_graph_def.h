/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_CORE_UTIL_EQUAL_GRAPH_DEF_H_
#define MACHINA_CORE_UTIL_EQUAL_GRAPH_DEF_H_

#include "machina/core/framework/graph_def_util.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/types.h"

namespace machina {

class GraphDef;
class NodeDef;

struct EqualGraphDefOptions {
  // Should internal attributes (attribute names that start with '_') be
  // ignored?
  bool ignore_internal_attrs = true;
};

// Determines if actual and expected are equal, ignoring versions and ordering
// of nodes, attrs, and control inputs.  If the GraphDefs are different and
// diff != nullptr, *diff is set to an explanation of the difference.  Note that
// we use node names to match up nodes between the graphs, and so the naming of
// nodes must be consistent.
bool EqualGraphDef(const GraphDef& actual, const GraphDef& expected,
                   string* diff, const EqualGraphDefOptions& options = {});

// Returns a hash of `gdef` that is consistent with EqualGraphDef. In other
// words, if two graph defs compare equal according to EqualGraphDef,
// GraphDefHash will return the same value for both of them when called
// with the same `options` that was used in the call to EqualGraphDef.
// Similarly to protobuf deterministic serialization, hash value is
// guaranteed to be stable only for a given binary. In particular, one should
// probably not persist the returned value.
uint64 GraphDefHash(const GraphDef& gdef,
                    const EqualGraphDefOptions& options = {});

// Determines if actual and expected are equal, ignoring: ordering of
// attrs, internal attributes (if set in `options`), and control inputs.
//
// If the NodeDefs are different and
// diff != nullptr, *diff is set to an explanation of the difference.
bool EqualNodeDef(const NodeDef& actual, const NodeDef& expected, string* diff,
                  const EqualGraphDefOptions& options = {});

// Returns a hash of `ndef` that is consistent with EqualNodeDef. In other
// words, if two node defs compare equal according to EqualNodeDef, NodeDefHash
// will return the same value for both of them when called with the same
// `options` that was used in the call to EqualNodeDef.
// Similarly to protobuf deterministic serialization, hash value is
// guaranteed to be stable only for a given binary. In particular, one should
// probably not persist the returned value.
uint64 NodeDefHash(const NodeDef& ndef,
                   const EqualGraphDefOptions& options = {});

// Determines if actual and expected are equal, ignoring ordering. If they're
// different and diff != nullptr, *diff is set to an explanation of the
// difference.
bool EqualRepeatedNodeDef(const protobuf::RepeatedPtrField<NodeDef>& actual,
                          const protobuf::RepeatedPtrField<NodeDef>& expected,
                          string* diff,
                          const EqualGraphDefOptions& options = {});

// Returns a hash of `ndefs` that is consistent with EqualRepeatedNodeDef.
// In other words, if two ndefs compare equal according to
// EqualRepeatedNodeDef, RepeatedNodeDefHash will return the same value for
// both of them when called with the same `options` that was used in
// the call to EqualRepeatedNodeDef.
// Similarly to protobuf deterministic serialization, hash value is
// guaranteed to be stable only for a given binary. In particular, one should
// probably not persist the returned value.
uint64 RepeatedNodeDefHash(const protobuf::RepeatedPtrField<NodeDef>& ndefs,
                           const EqualGraphDefOptions& options = {});

#define TF_EXPECT_GRAPH_EQ(expected, actual)            \
  do {                                                  \
    string diff;                                        \
    EXPECT_TRUE(EqualGraphDef(actual, expected, &diff)) \
        << diff << "\nExpected:\n"                      \
        << SummarizeGraphDef(expected) << "\nActual:\n" \
        << SummarizeGraphDef(actual);                   \
  } while (false)

}  // namespace machina

#endif  // MACHINA_CORE_UTIL_EQUAL_GRAPH_DEF_H_
