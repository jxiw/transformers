# coding=utf-8
# Copyright 2022 Junxiong Wang and The HuggingFace Inc. team. All rights reserved.
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
""" Flax S4BNBert model. """

from functools import partial
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax.nn.initializers import lecun_normal, uniform
from jax.numpy.linalg import eig, inv, matrix_power
from jax.scipy.signal import convolve

from .configuration_s4_bn_bert import S4BNBertConfig
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxMaskedLMOutput,
    FlaxCausalLMOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
)
from ...utils import add_start_docstrings
from ...utils import logging

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "s4_bn_bert"
_CONFIG_FOR_DOC = "S4BNBertConfig"
_TOKENIZER_FOR_DOC = "S4BNBertTokenizer"

S4_BN_BERT_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax
    Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`~S4BNBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the
            model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on
            GPUs) and `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see
            [`~FlaxPreTrainedModel.to_fp16`] and [`~FlaxPreTrainedModel.to_bf16`].
"""

S4_BN_BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`~S4BNBertTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`~S4BNBertTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for
            details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to
            the right for denoising pre-training following the paper.
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

S4_BN_BERT_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`~S4BNBertTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

S4_BN_BERT_DECODE_INPUTS_DOCSTRING = r"""
    Args:
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`~S4BNBertTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for
            details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to
            the right for denoising pre-training following the paper.
        encoder_outputs (`tuple(tuple(jnp.ndarray)`):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
            `attentions`) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`,
            *optional*) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        encoder_attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def non_circular_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
        Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return jnp.fft.irfft(out)[: u.shape[0]]


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
                jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)

    return init


@partial(jnp.vectorize, signature="(c),(),(c)->()")
def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum()


def K_gen_DPLR(Lambda, p, q, B, Ct, step, unmat=False):
    aterm = (Ct.conj().ravel(), q.conj().ravel())
    bterm = (B.ravel(), p.ravel())

    def gen(o):
        g = (2.0 / step) * ((1.0 - o) / (1.0 + o))
        c = 2.0 / (1.0 + o)

        def k(a):
            # Checkpoint this calculation for memory efficiency.
            if unmat:
                return jax.remat(cauchy_dot)(a, g, Lambda)
            else:
                return cauchy_dot(a, g, Lambda)

        k00 = k(aterm[0] * bterm[0])
        k01 = k(aterm[0] * bterm[1])
        k10 = k(aterm[1] * bterm[0])
        k11 = k(aterm[1] * bterm[1])
        return c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    return gen


def conv_from_gen(gen, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = jnp.exp((-2j * jnp.pi) * (jnp.arange(L) / L))
    atRoots = jax.vmap(gen)(Omega_L)
    # Inverse FFT
    out = jnp.fft.ifft(atRoots, L).reshape(L)
    return out.real


def discrete_DPLR(Lambda, p, q, B, Ct, step, L):
    N = Lambda.shape[0]
    A = jnp.diag(Lambda) - p[:, jnp.newaxis] @ q[:, jnp.newaxis].conj().T
    I = jnp.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
    qc = q.conj().T.reshape(1, -1)
    p2 = p.reshape(-1, 1)
    A1 = D - (D @ p2 * (1.0 / (1 + (qc @ D @ p2))) * qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


def make_HiPPO(N):
    def v(n, k):
        if n > k:
            return jnp.sqrt(2 * n + 1) * jnp.sqrt(2 * k + 1)
        elif n == k:
            return n + 1
        else:
            return 0

    # Do it slow so we don't mess it up :)
    mat = [[v(n, k) for k in range(1, N + 1)] for n in range(1, N + 1)]
    return -jnp.array(mat)


def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    p = 0.5 * jnp.sqrt(2 * jnp.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, jnp.newaxis] * q[jnp.newaxis, :]

    # Diagonalize to S to V \Lambda V^*
    Lambda, V = jax.jit(eig, backend="cpu")(S)
    # Lambda, V = eig(jax.device_put(S, device=jax.devices("cpu")[0]))
    # Lambda, V = eigh(S)
    return nhippo, Lambda, p, q, V


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


_, Lambda, p, q, V = make_NPLR_HiPPO(64)
Vc = V.conj().T
p = Vc @ p
q = Vc @ q.conj()
A = jnp.diag(Lambda) - p[:, jnp.newaxis] @ q[:, jnp.newaxis].conj().T


class S4Layer(nn.Module):
    A: jnp.DeviceArray
    Vc: jnp.DeviceArray
    p: jnp.DeviceArray
    q: jnp.DeviceArray
    Lambda: jnp.DeviceArray
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # Learned Parameters (Ct is complex!)
        self.Ct = self.param("Ct", lecun_normal(), (1, self.N, 2))
        self.Ct = self.Ct[..., 0] + 1j * self.Ct[..., 1]
        self.B = self.Vc @ self.param("B", lecun_normal(), (self.N, 1))
        self.D = self.param("D", uniform(), (1,))
        self.step = jnp.exp(self.param("log_step", log_step_initializer(), (1,)))

        if not self.decode:
            # CNN mode, compute kernel.
            K_gen = K_gen_DPLR(
                self.Lambda,
                self.p,
                self.q,
                self.B,
                self.Ct,
                self.step[0],
                unmat=self.l_max > 1000,
            )
            self.K = conv_from_gen(K_gen, self.l_max)

        else:
            # RNN mode, discretize
            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return discrete_DPLR(
                    self.Lambda,
                    self.p,
                    self.q,
                    self.B,
                    self.Ct,
                    self.step[0],
                    self.l_max,
                )

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", jnp.zeros, (self.N,), jnp.complex64
            )

    def __call__(self, u):
        # This is identical to SSM Layer
        if not self.decode:
            # CNN Mode
            return non_circular_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, jnp.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


class FlaxS4FeedForward(nn.Module):
    config: S4BNBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.c_fc = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]
        self.c_proj = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxS4BNLayer(nn.Module):
    config: S4BNBertConfig
    A: jnp.DeviceArray
    Vc: jnp.DeviceArray
    p: jnp.DeviceArray
    q: jnp.DeviceArray
    Lambda: jnp.DeviceArray
    decode: bool = False
    training: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_ssm = self.config.num_ssm
        self.max_seq_length = self.config.max_position_embeddings
        self.pre_norm = self.config.pre_norm
        self.BatchNorm = nn.BatchNorm(use_running_average=not self.training)
        self.fs4 = nn.vmap(
            S4Layer,
            in_axes=1,
            out_axes=1,
            variable_axes={"params": 1, "cache": 1, "prime": 1},
            split_rngs={"params": True},
        )
        self.bs4 = nn.vmap(
            S4Layer,
            in_axes=1,
            out_axes=1,
            variable_axes={"params": 1, "cache": 1, "prime": 1},
            split_rngs={"params": True},
        )
        self.feed_forward = FlaxS4FeedForward(self.config, dtype=self.dtype)

    @nn.compact
    def __call__(
            self,
            hidden_states,
            deterministic: bool = True
    ):

        @jax.vmap
        def apply_s4(hidden_states):
            # hidden_states should be max_seq_len * hidden size
            fs4_output = self.fs4(A=self.A, Vc=self.Vc, p=self.p,
                                  q=self.q, Lambda=self.Lambda, N=self.num_ssm,
                                  l_max=self.max_seq_length, decode=self.decode)(hidden_states)
            bs4_output = self.bs4(A=self.A, Vc=self.Vc, p=self.p,
                                  q=self.q, Lambda=self.Lambda, N=self.num_ssm,
                                  l_max=self.max_seq_length, decode=self.decode)(jnp.flip(hidden_states, axis=0))
            combine_output = fs4_output + jnp.flip(bs4_output, axis=0)
            output = self.feed_forward(combine_output, deterministic=deterministic)
            return output

        if self.pre_norm:
            hidden_states = hidden_states + apply_s4(self.BatchNorm(hidden_states))
        else:
            hidden_states = self.BatchNorm(hidden_states + apply_s4(hidden_states))

        return hidden_states


class FlaxS4BNLayerCollection(nn.Module):
    config: S4BNBertConfig
    training: bool = True
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxS4BNLayer(config=self.config, A=A, Vc=Vc, p=p, q=q, Lambda=Lambda, training=self.training, name=str(i),
                          dtype=self.dtype) for i in
            range(self.config.num_hidden_layers)
        ]

    def __call__(
            self,
            hidden_states,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = layer(
                hidden_states,
                deterministic=deterministic,
            )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states
        )


class FlaxS4BNEncoder(nn.Module):
    config: S4BNBertConfig
    training: bool = True
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxS4BNLayerCollection(self.config, dtype=self.dtype)

    def __call__(
            self,
            hidden_states,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxS4BNEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: S4BNBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        if self.config.token_type_embed:
            self.token_type_embeddings = nn.Embed(
                self.config.type_vocab_size,
                self.config.hidden_size,
                embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            )

    def __call__(self, input_ids, token_type_ids, deterministic: bool = True):
        # Embed
        hidden_states = self.word_embeddings(input_ids.astype("i4"))
        if self.config.token_type_embed:
            token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))
            # Sum all embeddings
            hidden_states = hidden_states + token_type_embeddings

        return hidden_states


class FlaxS4BNBertModule(nn.Module):
    config: S4BNBertConfig
    add_pooling_layer: bool = True
    training: bool = True
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.embeddings = FlaxS4BNEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxS4BNEncoder(self.config, training=self.training, dtype=self.dtype)

    # def _get_encoder_module(self):
    #     return self.encoder
    #
    # def _get_decoder_module(self):
    #     return self.decoder

    def __call__(
            self,
            input_ids,
            token_type_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):

        hidden_states = self.embeddings(
            input_ids, token_type_ids=token_type_ids, deterministic=deterministic
        )

        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pooled = jnp.mean(hidden_states, axis=-2) if self.add_pooling_layer else None
        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states
        )


class FlaxS4BNBertLMPredictionHead(nn.Module):
    config: S4BNBertConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)

        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        return hidden_states


class FlaxS4BNBertOnlyMLMHead(nn.Module):
    config: S4BNBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxS4BNBertLMPredictionHead(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states


class FlaxS4BNBertPreTrainedModel(FlaxPreTrainedModel):
    config_class = S4BNBertConfig
    base_model_prefix: str = "s4_bn_bert"
    module_class: nn.Module = None

    def __init__(
            self, config: S4BNBertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32,
            **kwargs
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        input_shape = (1, config.max_position_embeddings)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        params = self.module.init(
            rngs,
            input_ids,
            token_type_ids,
            return_dict=False
        )
        # print(params)
        # print("batch stats:", params['batch_stats'].shape)
        self.batch_stats = params['batch_stats']
        return params['params']

    # @add_start_docstrings_to_model_forward(S4_BN_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
            self,
            input_ids,
            token_type_ids=None,
            params: dict = None,
            batch_stats: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {
                "params": params or self.params,
                "batch_stats": batch_stats or self.batch_stats
            },
            jnp.array(input_ids, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            not train,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=["batch_stats"],
        )


@add_start_docstrings(
    "The bare S4BNBert Model transformer outputting raw hidden-states without any specific head on top.",
    S4_BN_BERT_START_DOCSTRING,
)
class FlaxS4BNBertModel(FlaxS4BNBertPreTrainedModel):
    config: S4BNBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    module_class = FlaxS4BNBertModule


class FlaxS4BNBertForMaskedLMModule(nn.Module):
    config: S4BNBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxS4BNBertModule(config=self.config, add_pooling_layer=False, dtype=self.dtype)
        self.cls = FlaxS4BNBertOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            token_type_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.s4_bert(
            input_ids,
            token_type_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.s4_bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states
        )


@add_start_docstrings("""S4BNBert Model with a `language modeling` head on top for MLM training. """,
                      S4_BN_BERT_START_DOCSTRING)
class FlaxS4BNBertForMaskedLM(FlaxS4BNBertPreTrainedModel):
    module_class = FlaxS4BNBertForMaskedLMModule


append_call_sample_docstring(
    FlaxS4BNBertForMaskedLM, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC
)


class FlaxS4BNBertForSequenceClassificationModule(nn.Module):
    config: S4BNBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxS4BNBertModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

    def __call__(
            self,
            input_ids,
            token_type_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.s4_bert(
            input_ids,
            token_type_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        if not return_dict:
            return (logits,) + outputs[2:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states
        )


@add_start_docstrings(
    """
    S4BNBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    S4_BN_BERT_START_DOCSTRING,
)
class FlaxS4BNBertForSequenceClassification(FlaxS4BNBertPreTrainedModel):
    module_class = FlaxS4BNBertForSequenceClassificationModule


append_call_sample_docstring(
    FlaxS4BNBertForSequenceClassification,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


class FlaxS4BNBertForTokenClassificationModule(nn.Module):
    config: S4BNBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxS4BNBertModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            token_type_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.s4_bert(
            input_ids,
            token_type_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states
        )


@add_start_docstrings(
    """
    S4BNBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    S4_BN_BERT_START_DOCSTRING,
)
class FlaxS4BNBertForTokenClassification(FlaxS4BNBertPreTrainedModel):
    module_class = FlaxS4BNBertForTokenClassificationModule


append_call_sample_docstring(
    FlaxS4BNBertForTokenClassification, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC
)


class FlaxS4BNBertForQuestionAnsweringModule(nn.Module):
    config: S4BNBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxS4BNBertModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            token_type_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.s4_bert(
            input_ids,
            token_type_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states
        )


@add_start_docstrings(
    """
    S4BNBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    S4_BN_BERT_START_DOCSTRING,
)
class FlaxS4BNBertForQuestionAnswering(FlaxS4BNBertPreTrainedModel):
    module_class = FlaxS4BNBertForQuestionAnsweringModule


append_call_sample_docstring(
    FlaxS4BNBertForQuestionAnswering,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)
