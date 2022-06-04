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
""" Flax S4BERT model. """

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

from .configuration_s4_bert import S4BertConfig
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxCausalLMOutput,
    FlaxMaskedLMOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...utils import logging

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "s4-base"
_CONFIG_FOR_DOC = "S4BertConfig"
_TOKENIZER_FOR_DOC = "S4BertTokenizer"
S4_BERT_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading, saving and converting weights from
    PyTorch models)

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module
    and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`~S4_BERTConfig`]): Model configuration class with all the parameters of the model.
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
S4_BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`~S4_BERTConfiTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

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
    # Make HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    p = 0.5 * jnp.sqrt(2 * jnp.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, jnp.newaxis] * q[jnp.newaxis, :]

    # Diagonalize to S to V \Lambda V^*
    Lambda, V = jax.jit(eig, backend="cpu")(S)
    return nhippo, Lambda, p, q, V

def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)

# only run once
_, Lambda, p, q, V = make_NPLR_HiPPO(N)
Vc = V.conj().T
p = Vc @ p
q = Vc @ q.conj()
A = jnp.diag(Lambda) - p[:, jnp.newaxis] @ q[:, jnp.newaxis].conj().T

def vc_initializer(N):
    def init(key):
        return Vc
    return init

def p_initializer(N):
    def init(key):
        return p
    return init

def q_initializer(N):
    def init(key):
        return q
    return init

def lambda_initializer(N):
    def init(key):
        return Lambda
    return init

class S4Kernel(nn.Module):
    N: int
    l_max: int

    def setup(self):
        # trained params
        self.p = self.param("p", p_initializer(self.N))
        self.q = self.param("q", q_initializer(self.N))
        self.Vc = self.param("Vc", vc_initializer(self.N))
        self.Lambda = self.param("Lambda", lambda_initializer(self.N))
        # previous params
        self.Ct = self.param("Ct", lecun_normal(), (1, self.N, 2))
        self.Ct = self.Ct[..., 0] + 1j * self.Ct[..., 1]
        self.B = self.Vc @ self.param("B", lecun_normal(), (self.N, 1))
        self.step = jnp.exp(self.param("log_step", log_step_initializer(), (1,)))

    def __call__(self):
        K_gen = K_gen_DPLR(
            self.Lambda,
            self.p,
            self.q,
            self.B,
            self.Ct,
            self.step[0],
            unmat=self.l_max > 4096,
        )
        return conv_from_gen(K_gen, self.l_max)

class S4Layer(nn.Module):
    N: int
    l_max: int

    def setup(self):
        # Learned Parameters (Ct is complex!)
        self.D = self.param("D", uniform(), (1,))
        self.s4_kernel = S4Kernel(self.N, self.l_max)
        if self.decode:
            # RNN mode, discretize
            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return discrete_DPLR(
                    self.s4_kernel.Lambda,
                    self.s4_kernel.p,
                    self.s4_kernel.q,
                    self.s4_kernel.B,
                    self.s4_kernel.Ct,
                    self.s4_kernel.step[0],
                    self.s4_kernel.l_max,
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
            K = self.s4_kernel()
            return non_circular_convolution(u, K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, jnp.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u

class FlaxS4BertIntermediate(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class FlaxS4BertOutput(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states

class FlaxS4Layer(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.fs4 = S4Layer(
            N=self.config.num_ssm, l_max=self.config.max_position_embeddings
        )
        self.bs4 = S4Layer(
            N=self.config.num_ssm, l_max=self.config.max_position_embeddings
        )
        self.intermediate = FlaxS4BertIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxS4BertOutput(self.config, dtype=self.dtype)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(
            self,
            hidden_states,
            deterministic: bool = True
    ):
        @jax.vmap
        def apply_s4(hidden_states):
            # the hidden states shape should be [sentence length x hidden dim]
            fs4_output = jax.vmap(self.fs4.__call__, in_axes=1, out_axes=1)(hidden_states)
            bs4_output = jax.vmap(self.bs4.__call__, in_axes=1, out_axes=1)(jnp.flip(hidden_states, axis=0))
            # we sum states
            # s4_output = fs4_1_output + jnp.flip(bs4_1_output, axis=0)
            # we sum states using residual connection like transfomer
            # s4_output = hidden_states + fs4_1_output + jnp.flip(bs4_1_output, axis=0)
            # instead of sum, we concat states
            bs4_output = jnp.flip(bs4_output, axis=0)
            hidden_output = jnp.concatenate([fs4_output, bs4_output], axis=-1)
            return hidden_output

        if self.config.pre_norm:
            s4_hidden_states = hidden_states
            s4_hidden_states = self.LayerNorm(s4_hidden_states)
            s4_hidden_states = apply_s4(s4_hidden_states)
            s4_hidden_states = self.intermediate(s4_hidden_states)
            s4_hidden_states = self.output(s4_hidden_states, deterministic=deterministic)
            outputs = hidden_states + s4_hidden_states
        else:
            s4_hidden_states = hidden_states
            s4_hidden_states = apply_s4(s4_hidden_states)
            s4_hidden_states = self.intermediate(s4_hidden_states)
            s4_hidden_states = self.output(s4_hidden_states, deterministic=deterministic)
            outputs = self.LayerNorm(hidden_states + s4_hidden_states)

        return outputs

class FlaxS4LayerCollection(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # generate S4 matrix to train
        # here we use those global variables, A, Vc, p, q, Lambda
        self.layers = [
            FlaxS4Layer(config=self.config, name=str(i), dtype=self.dtype) for i in
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

        for i, layer in enumerate(self.layers):
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


class FlaxS4Encoder(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxS4LayerCollection(self.config, dtype=self.dtype)

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


class FlaxS4Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, deterministic: bool = True):
        # Embed
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxS4Pooler(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        avg_hidden_state = jnp.mean(hidden_states, axis=-2)
        avg_hidden_state = self.dense(avg_hidden_state)
        return nn.tanh(avg_hidden_state)


class FlaxS4BertModule(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxS4Embeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxS4Encoder(self.config, dtype=self.dtype)
        self.pooler = FlaxS4Pooler(self.config, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            token_type_ids: Optional[np.ndarray] = None,
            position_ids: Optional[np.ndarray] = None,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):

        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # make sure `position_ids` is correctly initialized when not passed
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=0)
        # def vectorize_s4(s_input_ids, s_token_type_ids, s_position_ids):
        #     # input should be
        #     hidden_states = self.embeddings(
        #         input_ids=s_input_ids, token_type_ids=s_token_type_ids, position_ids=s_position_ids, deterministic=deterministic
        #     )
        #
        #     outputs = self.encoder(
        #         hidden_states,
        #         deterministic=deterministic,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        #
        #     return outputs
        #
        # outputs = vectorize_s4(input_ids, token_type_ids, position_ids)

        # keep same as bert
        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, deterministic=deterministic
        )
        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None
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


class FlaxBertPredictionHeadTransform(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.LayerNorm(hidden_states)


class FlaxS4BertLMPredictionHead(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transform = FlaxBertPredictionHeadTransform(self.config, dtype=self.dtype)
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.transform(hidden_states)

        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)

        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        return hidden_states


class FlaxS4BertOnlyMLMHead(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxS4BertLMPredictionHead(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states


class FlaxS4BertPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = S4BertConfig
    base_model_prefix = "s4_bert"
    module_class: nn.Module = None

    def __init__(
            self, config: S4BertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32,
            **kwargs
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        input_shape = (1, config.max_position_embeddings)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        return self.module.init(
            rngs, input_ids, token_type_ids, position_ids, return_dict=False
        )["params"]

    @add_start_docstrings_to_model_forward(S4_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
            self,
            input_ids,
            token_type_ids=None,
            position_ids=None,
            params: dict = None,
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

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


add_start_docstrings(
    "The bare S4Bert Model transformer outputting raw hidden-states without any specific head on top.",
    S4_BERT_START_DOCSTRING,
)


class FlaxS4BertModel(FlaxS4BertPreTrainedModel):
    module_class = FlaxS4BertModule


class FlaxS4BertForMaskedLMModule(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxS4BertModule(config=self.config, add_pooling_layer=False, dtype=self.dtype)
        self.cls = FlaxS4BertOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            token_type_ids,
            position_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.s4_bert(
            input_ids,
            token_type_ids,
            position_ids,
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


@add_start_docstrings("""S4Bert Model with a `language modeling` head on top for MLM training. """,
                      S4_BERT_START_DOCSTRING)
class FlaxS4BertForMaskedLM(FlaxS4BertPreTrainedModel):
    module_class = FlaxS4BertForMaskedLMModule


append_call_sample_docstring(
    FlaxS4BertForMaskedLM, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC
)


# class FlaxS4BertForCausalLMModule(nn.Module):
#     config: S4BertConfig
#     dtype: jnp.dtype = jnp.float32
#
#     def setup(self):
#         self.s4_bert = FlaxS4BertModule(config=self.config, add_pooling_layer=False, dtype=self.dtype)
#         self.cls = FlaxS4BertOnlyMLMHead(config=self.config, dtype=self.dtype)
#
#     def __call__(
#         self,
#         input_ids,
#         attention_mask,
#         token_type_ids,
#         position_ids,
#         head_mask,
#         deterministic: bool = True,
#         output_attentions: bool = False,
#         output_hidden_states: bool = False,
#         return_dict: bool = True,
#     ):
#         # Model
#         outputs = self.s4_bert(
#             input_ids,
#             token_type_ids,
#             deterministic=deterministic,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         hidden_states = outputs[0]
#         if self.config.tie_word_embeddings:
#             shared_embedding = self.s4_bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
#         else:
#             shared_embedding = None
#
#         # Compute the prediction scores
#         logits = self.cls(hidden_states, shared_embedding=shared_embedding)
#
#         if not return_dict:
#             return (logits,) + outputs[1:]
#
#         return FlaxCausalLMOutput(
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
#
#
# @add_start_docstrings("""S4Bart Model with a `language modeling` head on top for CLM training. """, S4_BERT_START_DOCSTRING)
# class FlaxS4BertForCausalLM(FlaxS4BertPreTrainedModel):
#     module_class = FlaxS4BertForCausalLMModule
#
#
# append_call_sample_docstring(
#     FlaxS4BertForCausalLM, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutput, _CONFIG_FOR_DOC
# )

class FlaxS4BertForSequenceClassificationModule(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxS4BertModule(config=self.config, dtype=self.dtype)
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

    def __call__(
            self,
            input_ids,
            token_type_ids,
            position_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.s4_bert(
            input_ids,
            token_type_ids,
            position_ids,
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
    S4Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    S4_BERT_START_DOCSTRING,
)
class FlaxS4BertForSequenceClassification(FlaxS4BertPreTrainedModel):
    module_class = FlaxS4BertForSequenceClassificationModule


append_call_sample_docstring(
    FlaxS4BertForSequenceClassification,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


# class FlaxS4BertForMultipleChoiceModule(nn.Module):
#     config: S4BertConfig
#     dtype: jnp.dtype = jnp.float32
#
#     def setup(self):
#         self.s4_bert = FlaxS4BertModule(config=self.config, dtype=self.dtype)
#         self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
#         self.classifier = nn.Dense(1, dtype=self.dtype)
#
#     def __call__(
#             self,
#             input_ids,
#             token_type_ids,
#             deterministic: bool = True,
#             output_hidden_states: bool = False,
#             return_dict: bool = True,
#     ):
#         num_choices = input_ids.shape[1]
#         input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
#         token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
#         print("input_ids sss:", input_ids.shape)
#         print("token_type_ids sss:", token_type_ids.shape)
#
#         # Model
#         outputs = self.s4_bert(
#             input_ids,
#             token_type_ids,
#             deterministic=deterministic,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output, deterministic=deterministic)
#         logits = self.classifier(pooled_output)
#
#         reshaped_logits = logits.reshape(-1, num_choices)
#
#         if not return_dict:
#             return (reshaped_logits,) + outputs[2:]
#
#         return FlaxMultipleChoiceModelOutput(
#             logits=reshaped_logits,
#             hidden_states=outputs.hidden_states,
#         )
#
#
# @add_start_docstrings(
#     """
#     S4Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
#     softmax) e.g. for RocStories/SWAG tasks.
#     """,
#     S4_BERT_START_DOCSTRING,
# )
# class FlaxS4BertForMultipleChoice(FlaxS4BertPreTrainedModel):
#     module_class = FlaxS4BertForMultipleChoiceModule


# overwrite_call_docstring(
#     FlaxS4BertForMultipleChoice, S4_BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
# )
# append_call_sample_docstring(
#     FlaxS4BertForMultipleChoice, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxMultipleChoiceModelOutput, _CONFIG_FOR_DOC
# )


class FlaxS4BertForTokenClassificationModule(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxS4BertModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
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
    S4Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    S4_BERT_START_DOCSTRING,
)
class FlaxS4BertForTokenClassification(FlaxS4BertPreTrainedModel):
    module_class = FlaxS4BertForTokenClassificationModule


append_call_sample_docstring(
    FlaxS4BertForTokenClassification, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC
)


class FlaxS4BertForQuestionAnsweringModule(nn.Module):
    config: S4BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxS4BertModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
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
    S4Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    S4_BERT_START_DOCSTRING,
)
class FlaxS4BertForQuestionAnswering(FlaxS4BertPreTrainedModel):
    module_class = FlaxS4BertForQuestionAnsweringModule


append_call_sample_docstring(
    FlaxS4BertForQuestionAnswering,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)
