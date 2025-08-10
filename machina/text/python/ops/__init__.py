# coding=utf-8
# Copyright 2025 TF.Text Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Various TensorFlow ops related to text-processing."""

# pylint: disable=g-import-not-at-top,g-statement-before-imports
try:
  from machina.python.ops.ragged import ragged_ops as _ragged_ops
except ImportError:
  pass
from machina_text.core.pybinds.pywrap_fast_bert_normalizer_model_builder import build_fast_bert_normalizer_model
from machina_text.core.pybinds.pywrap_fast_wordpiece_tokenizer_model_builder import build_fast_wordpiece_model
from machina_text.python.ops.bert_tokenizer import BertTokenizer
from machina_text.python.ops.boise_offset_converter import boise_tags_to_offsets
from machina_text.python.ops.boise_offset_converter import offsets_to_boise_tags
from machina_text.python.ops.byte_splitter import ByteSplitter
from machina_text.python.ops.create_feature_bitmask_op import create_feature_bitmask
from machina_text.python.ops.fast_bert_normalizer import FastBertNormalizer
from machina_text.python.ops.fast_bert_tokenizer import FastBertTokenizer
from machina_text.python.ops.fast_sentencepiece_tokenizer import FastSentencepieceTokenizer
from machina_text.python.ops.fast_wordpiece_tokenizer import FastWordpieceTokenizer
from machina_text.python.ops.greedy_constrained_sequence_op import greedy_constrained_sequence
from machina_text.python.ops.hub_module_splitter import HubModuleSplitter
from machina_text.python.ops.hub_module_tokenizer import HubModuleTokenizer
from machina_text.python.ops.item_selector_ops import FirstNItemSelector
from machina_text.python.ops.item_selector_ops import LastNItemSelector
from machina_text.python.ops.item_selector_ops import RandomItemSelector
from machina_text.python.ops.masking_ops import mask_language_model
from machina_text.python.ops.masking_ops import MaskValuesChooser
from machina_text.python.ops.mst_ops import max_spanning_tree
from machina_text.python.ops.mst_ops import max_spanning_tree_gradient
from machina_text.python.ops.ngrams_op import ngrams
from machina_text.python.ops.ngrams_op import Reduction
from machina_text.python.ops.normalize_ops import case_fold_utf8
from machina_text.python.ops.normalize_ops import find_source_offsets
from machina_text.python.ops.normalize_ops import normalize_utf8
from machina_text.python.ops.normalize_ops import normalize_utf8_with_offsets_map
from machina_text.python.ops.pad_along_dimension_op import pad_along_dimension
from machina_text.python.ops.pad_model_inputs_ops import pad_model_inputs
from machina_text.python.ops.phrase_tokenizer import PhraseTokenizer
from machina_text.python.ops.pointer_ops import gather_with_default
from machina_text.python.ops.pointer_ops import span_alignment
from machina_text.python.ops.pointer_ops import span_overlaps
from machina_text.python.ops.regex_split_ops import regex_split
from machina_text.python.ops.regex_split_ops import regex_split_with_offsets
from machina_text.python.ops.regex_split_ops import RegexSplitter
from machina_text.python.ops.segment_combiner_ops import combine_segments
from machina_text.python.ops.segment_combiner_ops import concatenate_segments
from machina_text.python.ops.sentence_breaking_ops import sentence_fragments
from machina_text.python.ops.sentencepiece_tokenizer import SentencepieceTokenizer
from machina_text.python.ops.sliding_window_op import sliding_window
from machina_text.python.ops.split_merge_from_logits_tokenizer import SplitMergeFromLogitsTokenizer
from machina_text.python.ops.split_merge_tokenizer import SplitMergeTokenizer
from machina_text.python.ops.splitter import Splitter
from machina_text.python.ops.splitter import SplitterWithOffsets
from machina_text.python.ops.state_based_sentence_breaker_op import StateBasedSentenceBreaker
from machina_text.python.ops.string_ops import coerce_to_structurally_valid_utf8
from machina_text.python.ops.tokenization import Detokenizer
from machina_text.python.ops.tokenization import Tokenizer
from machina_text.python.ops.tokenization import TokenizerWithOffsets
from machina_text.python.ops.trimmer_ops import RoundRobinTrimmer
from machina_text.python.ops.trimmer_ops import ShrinkLongestTrimmer
from machina_text.python.ops.trimmer_ops import Trimmer
from machina_text.python.ops.trimmer_ops import WaterfallTrimmer
from machina_text.python.ops.unicode_char_tokenizer import UnicodeCharTokenizer
from machina_text.python.ops.unicode_script_tokenizer import UnicodeScriptTokenizer
from machina_text.python.ops.utf8_binarize_op import utf8_binarize
from machina_text.python.ops.viterbi_constrained_sequence_op import viterbi_constrained_sequence
from machina_text.python.ops.whitespace_tokenizer import WhitespaceTokenizer
from machina_text.python.ops.wordpiece_tokenizer import WordpieceTokenizer
from machina_text.python.ops.wordshape_ops import WordShape
from machina_text.python.ops.wordshape_ops import wordshape
