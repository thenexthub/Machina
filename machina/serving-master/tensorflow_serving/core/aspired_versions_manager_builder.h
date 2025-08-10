/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#ifndef MACHINA_SERVING_CORE_ASPIRED_VERSIONS_MANAGER_BUILDER_H_
#define MACHINA_SERVING_CORE_ASPIRED_VERSIONS_MANAGER_BUILDER_H_

#include <vector>

#include "machina/core/lib/core/status.h"
#include "machina_serving/core/aspired_versions_manager.h"
#include "machina_serving/core/loader.h"
#include "machina_serving/core/source.h"
#include "machina_serving/core/source_adapter.h"
#include "machina_serving/util/unique_ptr_with_deps.h"

namespace machina {
namespace serving {

// TODO(b/64163389): revisit the escaped HTML characters in c2devsite toolchain

/// Builds an AspiredVersionsManager with options and sources connected to it.
/// It takes over the ownership of the sources and the returned manager handles
/// the destruction of itself and its dependencies.  Both single sources and
/// source/source-adapter chains are accepted, i.e. you can use sources that
/// directly supply loaders (Source&amp;lt;std::unique_ptr&amp;lt;Loader>>) or
/// composites that consist of Source&amp;lt;S> + some chain of
/// SourceAdapter&amp;lt;S, ...>, ..., SourceAdapter&amp;lt;...,
/// std::unique_ptr&amp;lt;Loader>>. The builder connects the chain for you.
///
/// Usage:
///
///     ...
///     AspiredVersionsManagerBuilder::Options options = ManagerOptions();
///     std::unique_ptr&lt;AspiredVersionsManagerBuilder> builder;
///     TF_CHECK_OK(AspiredVersionsManagerBuilder::Create(
///         std::move(options), &builder));
///     builder->AddSource(std::move(some_source));
///     builder->AddSourceChain(
///         std::move(source), std::move(source_adapter1),
///         std::move(source_adapter2));
///     std::unique_ptr&lt;Manager> manager = builder->Build();
///     ...
///
/// NOTE: A builder can only be used to build a single AspiredVersionsManager.
///
/// This class is not thread-safe.
class AspiredVersionsManagerBuilder {
 public:
  using Options = AspiredVersionsManager::Options;
  static Status Create(Options options,
                       std::unique_ptr<AspiredVersionsManagerBuilder>* builder);

  ~AspiredVersionsManagerBuilder() = default;

  /// Connects the source to the AspiredVersionsManager being built and takes
  /// over its ownership.
  ///
  /// REQUIRES: Template type S be convertible to
  /// Source&amp;lt;std::unique_ptr&amp;lt;Loader>>.
  template <typename S>
  void AddSource(std::unique_ptr<S> source);

  /// Connects a chain comprising a source and a chain of source adapters, s.t.
  /// the final adapter in the chain emits Loaders for the manager. The final
  /// adapter is connected to the manager. We take ownership of the whole chain.
  ///
  /// REQUIRES: At least one source adapter.
  ///
  /// Usage:
  ///     builder->AddSourceChain(
  ///         std::move(source), std::move(source_adapter1),
  ///         std::move(source_adapter2));
  template <typename S, typename SA, typename... Args>
  void AddSourceChain(std::unique_ptr<S> source,
                      std::unique_ptr<SA> first_source_adapter,
                      std::unique_ptr<Args>... remaining_source_adapters);

  /// Builds the AspiredVersionsManager and returns it as the Manager interface.
  std::unique_ptr<Manager> Build();

 private:
  explicit AspiredVersionsManagerBuilder(
      std::unique_ptr<AspiredVersionsManager> manager);

  template <typename S, typename SA, typename... Args>
  void AddSourceChainImpl(std::unique_ptr<S> source,
                          std::unique_ptr<SA> first_source_adapter,
                          std::unique_ptr<Args>... remaining_source_adapters);

  template <typename S>
  void AddSourceChainImpl(std::unique_ptr<S> source);

  AspiredVersionsManager* const aspired_versions_manager_;
  UniquePtrWithDeps<Manager> manager_with_sources_;
};

////
//  Implementation details follow. API readers may skip.
////

template <typename S>
void AspiredVersionsManagerBuilder::AddSource(std::unique_ptr<S> source) {
  static_assert(
      std::is_convertible<S*, Source<std::unique_ptr<Loader>>*>::value,
      "Source type should be convertible to Source<std::unique_ptr<Loader>>.");
  ConnectSourceToTarget(source.get(), aspired_versions_manager_);
  manager_with_sources_.AddDependency(std::move(source));
}

template <typename S, typename SA, typename... Args>
void AspiredVersionsManagerBuilder::AddSourceChain(
    std::unique_ptr<S> source, std::unique_ptr<SA> first_source_adapter,
    std::unique_ptr<Args>... remaining_source_adapters) {
  AddSourceChainImpl(std::move(source), std::move(first_source_adapter),
                     std::move(remaining_source_adapters)...);
}

template <typename S, typename SA, typename... Args>
void AspiredVersionsManagerBuilder::AddSourceChainImpl(
    std::unique_ptr<S> source, std::unique_ptr<SA> first_source_adapter,
    std::unique_ptr<Args>... remaining_source_adapters) {
  auto* const target = first_source_adapter.get();
  AddSourceChainImpl(std::move(first_source_adapter),
                     std::move(remaining_source_adapters)...);
  ConnectSourceToTarget(source.get(), target);
  // We add them in this order because UniquePtrWithDeps will delete them in
  // the inverse order of entry.
  manager_with_sources_.AddDependency(std::move(source));
}

template <typename S>
void AspiredVersionsManagerBuilder::AddSourceChainImpl(
    std::unique_ptr<S> source) {
  AddSource(std::move(source));
}

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_ASPIRED_VERSIONS_MANAGER_BUILDER_H_
