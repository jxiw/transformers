# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

# rely on isort to merge the imports
from ...utils import _LazyModule, is_tokenizers_available
from ...utils import is_flax_available


_import_structure = {
    "configuration_s4_bert": ["S4_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "S4BertConfig"],
    "tokenization_s4_bert": ["S4BertTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_s4_bert_fast"] = ["S4BertTokenizerFast"]

if is_flax_available():
    _import_structure["modeling_flax_s4_bert"] = [
        "FlaxS4BertForMaskedLM",
        # "FlaxS4BertForMultipleChoice",
        "FlaxS4BertForQuestionAnswering",
        "FlaxS4BertForSequenceClassification",
        "FlaxS4BertForTokenClassification",
        "FlaxS4BertLayer",
        "FlaxS4BertModel",
        "FlaxS4BertPreTrainedModel",
    ]




if TYPE_CHECKING:
    from .configuration_s4_bert import S4_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, S4BertConfig
    from .tokenization_s4_bert import S4BertTokenizer

    if is_tokenizers_available():
        from .tokenization_s4_bert_fast import S4BertTokenizerFast

    if is_flax_available():
        from .modeling_s4_bert import (
            FlaxS4BertForMaskedLM,
            FlaxS4BertForQuestionAnswering,
            FlaxS4BertForSequenceClassification,
            FlaxS4BertForTokenClassification,
            FlaxS4BertLayer,
            FlaxS4BertModel,
            FlaxS4BertPreTrainedModel,
        )



else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
