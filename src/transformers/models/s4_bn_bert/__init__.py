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
    "configuration_s4_bn_bert": ["S4_BN_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "S4BNBertConfig"],
    "tokenization_s4_bn_bert": ["S4BNBertTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_s4_bn_bert_fast"] = ["S4BNBertTokenizerFast"]

if is_flax_available():
    _import_structure["modeling_flax_s4_bn_bert"] = [
        "FlaxS4BNBertForMaskedLM",
        "FlaxS4BNBertForQuestionAnswering",
        "FlaxS4BNBertForSequenceClassification",
        "FlaxS4BNBertModel",
        "FlaxS4BNBertPreTrainedModel",
    ]




if TYPE_CHECKING:
    from .configuration_s4_bn_bert import S4_BN_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, S4BNBertConfig
    from .tokenization_s4_bn_bert import S4BNBertTokenizer

    if is_tokenizers_available():
        from .tokenization_s4_bn_bert_fast import S4BNBertTokenizerFast

    if is_flax_available():
        from .modeling_s4_bn_bert import (
            FlaxS4BNBertForQuestionAnswering,
            FlaxS4BNBertForSequenceClassification,
            FlaxS4BNBertModel,
            FlaxS4BNBertPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
