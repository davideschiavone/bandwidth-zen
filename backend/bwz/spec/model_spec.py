"""Model specs: parametric transformers, explicit-layer CNNs, and custom op lists.

Schema reference: ``docs/SCHEMA.md`` §2; source schema: ``PROMPT.md`` §4.2.

The three flavours are a discriminated union on ``family``. Whichever flavour is
used, :func:`load_model` returns something with an ``id``, a ``name`` and a
``parameter_count()``; everything else is family-specific and is expanded into a
graph at M2.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from bwz.spec.base import Identifier, SourceUrl, SpecModel
from bwz.spec.quantities import Bytes, Flops


class ModelFamily(StrEnum):
    """Which builder in ``graph/`` expands this spec."""

    TRANSFORMER_DECODER = "transformer_decoder"
    TRANSFORMER_ENCODER = "transformer_encoder"
    CNN = "cnn"
    CUSTOM = "custom"


class NormType(StrEnum):
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    BATCHNORM = "batchnorm"
    NONE = "none"


class FFNType(StrEnum):
    """Feed-forward shape. The gated variants carry three weight matrices, not two."""

    RELU = "relu"
    GELU = "gelu"
    SWIGLU = "swiglu"
    GEGLU = "geglu"

    @property
    def is_gated(self) -> bool:
        return self in (FFNType.SWIGLU, FFNType.GEGLU)

    @property
    def n_matrices(self) -> int:
        """3 for gated FFNs (gate, up, down), 2 otherwise (up, down)."""
        return 3 if self.is_gated else 2


class PositionalType(StrEnum):
    LEARNED = "learned"
    ROPE = "rope"
    ALIBI = "alibi"
    NONE = "none"


class TransformerParams(SpecModel):
    """Hyperparameters of a parametric transformer, named as in a HF ``config.json``."""

    layers: int = Field(gt=0)
    hidden: int = Field(gt=0)
    heads: int = Field(gt=0)
    kv_heads: int | None = Field(default=None, gt=0, description="GQA/MQA; defaults to heads")
    head_dim: int | None = Field(
        default=None, gt=0, description="Defaults to hidden // heads; Gemma-3 sets it explicitly"
    )
    ffn_hidden: int = Field(gt=0)
    ffn_type: FFNType
    vocab: int = Field(gt=0)
    max_context: int = Field(gt=0)
    norm: NormType = NormType.RMSNORM
    positional: PositionalType = PositionalType.ROPE
    tie_embeddings: bool = False
    sliding_window: int | None = Field(default=None, gt=0)
    sliding_window_pattern: int | None = Field(
        default=None, gt=0, description="1 global layer every N layers (Gemma-3 uses 6)"
    )

    @model_validator(mode="after")
    def _check_shapes(self) -> TransformerParams:
        if self.head_dim is None and self.hidden % self.heads != 0:
            raise ValueError(
                f"hidden ({self.hidden}) is not divisible by heads ({self.heads}) and head_dim is "
                f"not set; either set head_dim explicitly or choose a head count dividing "
                f"{self.hidden}"
            )
        if self.effective_kv_heads > self.heads:
            raise ValueError(
                f"kv_heads ({self.effective_kv_heads}) must not exceed heads ({self.heads}); "
                f"GQA groups query heads onto a smaller number of key/value heads"
            )
        if self.heads % self.effective_kv_heads != 0:
            raise ValueError(
                f"heads ({self.heads}) is not divisible by kv_heads ({self.effective_kv_heads}); "
                f"GQA requires an integer number of query heads per key/value head"
            )
        if self.sliding_window_pattern is not None and self.sliding_window is None:
            raise ValueError(
                "sliding_window_pattern is set but sliding_window is not; a pattern without a "
                "window size has no meaning"
            )
        return self

    @property
    def effective_kv_heads(self) -> int:
        """``kv_heads`` if set, otherwise ``heads`` (i.e. plain multi-head attention)."""
        return self.kv_heads if self.kv_heads is not None else self.heads

    @property
    def effective_head_dim(self) -> int:
        """``head_dim`` if set, otherwise ``hidden // heads``."""
        return self.head_dim if self.head_dim is not None else self.hidden // self.heads

    @property
    def q_width(self) -> int:
        """Output width of the Q projection: ``heads x head_dim``. Equals ``hidden`` only when
        ``head_dim`` is implicit — Gemma-3-4B has 8 x 256 = 2048 against a hidden of 2560."""
        return self.heads * self.effective_head_dim

    @property
    def kv_width(self) -> int:
        """Output width of the K (and V) projection: ``kv_heads x head_dim``."""
        return self.effective_kv_heads * self.effective_head_dim

    # -- derived parameter counts -------------------------------------------

    def attention_params_per_layer(self) -> int:
        """Q, K, V, O projection weights for one layer (biases excluded — no modern
        decoder in the shipped profiles uses them)."""
        q = self.hidden * self.q_width
        k = self.hidden * self.kv_width
        v = self.hidden * self.kv_width
        o = self.q_width * self.hidden
        return q + k + v + o

    def ffn_params_per_layer(self) -> int:
        """``n_matrices x hidden x ffn_hidden`` — three matrices for a gated FFN."""
        return self.ffn_type.n_matrices * self.hidden * self.ffn_hidden

    def norm_params_per_layer(self) -> int:
        """Two norms per block; layernorm carries a bias as well as a scale."""
        per_norm = {
            NormType.RMSNORM: self.hidden,
            NormType.LAYERNORM: 2 * self.hidden,
            NormType.BATCHNORM: 2 * self.hidden,
            NormType.NONE: 0,
        }[self.norm]
        return 2 * per_norm

    def params_per_layer(self) -> int:
        return (
            self.attention_params_per_layer()
            + self.ffn_params_per_layer()
            + self.norm_params_per_layer()
        )

    def embedding_params(self) -> int:
        """Token embeddings, plus a separate output head when embeddings are untied."""
        table = self.vocab * self.hidden
        return table if self.tie_embeddings else 2 * table

    def parameter_count(self) -> int:
        """Total weight count, derived closed-form from the hyperparameters.

        Formula (docs/MODEL.md): ``layers x (attn + ffn + norm) + embeddings +
        final_norm``. Cross-checked at M2 against the graph builder's summed
        weight tensors — two independent derivations agreeing is a stronger test
        than either alone.

        Worked example (Llama-3-8B): attention ``4096x4096 + 2x4096x1024 +
        4096x4096 = 41.94 M``; FFN ``3x4096x14336 = 176.16 M``; norms ``2x4096``;
        so ``32 x 218.11 M = 6.980 G``, plus untied embeddings
        ``2 x 128256 x 4096 = 1.051 G``, total **8.030 G**.
        """
        final_norm = self.norm_params_per_layer() // 2
        return self.layers * self.params_per_layer() + self.embedding_params() + final_norm

    def kv_cache_bytes(self, context: int, bytes_per_element: float, batch: int = 1) -> float:
        """KV-cache footprint: ``layers x 2 x kv_width x context x bytes x batch``.

        The 2 is K and V. Worked example (Llama-3-8B, 8k context, fp16, batch 1):
        ``32 x 2 x 1024 x 8192 x 2 = 1.074e9`` bytes = 1.07 GB.
        """
        return self.layers * 2 * self.kv_width * context * bytes_per_element * batch


class _ModelBase(SpecModel):
    """Fields common to every model flavour."""

    id: Identifier
    name: str = Field(min_length=1)
    source_url: SourceUrl | None = None
    hypothetical: bool = False
    estimates: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_provenance(self) -> _ModelBase:
        if not self.hypothetical and self.source_url is None:
            raise ValueError(
                f"model {self.id!r}: source_url is required (CLAUDE.md: never commit profile YAML "
                f"without a source_url). Cite the HF config.json or the paper. If this profile is "
                f"a synthetic variant rather than a published model, set 'hypothetical: true'."
            )
        return self


class TransformerSpec(_ModelBase):
    """A parametric transformer, decoder or encoder."""

    family: Literal[ModelFamily.TRANSFORMER_DECODER, ModelFamily.TRANSFORMER_ENCODER]
    params: TransformerParams
    declared_params: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Headline parameter count, when the profile is a size preset whose hyperparameters "
            "do not sum to it. Drives W = params x w_bytes; the divergence from the derived "
            "count is reported as an assumption (docs/CORRECTIONS.md D8)."
        ),
    )
    preset_of: Identifier | None = Field(
        default=None, description="Base profile this preset scales, e.g. gemma3_4b"
    )

    @model_validator(mode="after")
    def _check_preset(self) -> TransformerSpec:
        if self.preset_of is not None and self.declared_params is None:
            raise ValueError(
                f"model {self.id!r}: preset_of is set but declared_params is not; a size preset "
                f"exists precisely to override the headline parameter count"
            )
        return self

    @property
    def effective_params(self) -> TransformerParams:
        """The hyperparameters that are actually built.

        For a size preset, ``declared_params`` is realised by **scaling the layer
        count** — the dimension that genuinely varies within a model family, and
        the only one that can be changed without inventing a new architecture.
        Widths and vocabulary stay at the family's values.

        The realisation is never exact, because layers are integers and the
        embedding table does not scale at all. Gemma-3-4B's tied 262208x2560
        table is 671 M parameters, so a 1 B preset of that family is 70%
        embedding and has room for only 3 blocks — a genuinely degenerate model,
        and the reason the real ``gemma3_1b`` profile is the better 1 B point.
        :meth:`declared_vs_derived_error` reports the residual.
        """
        if self.declared_params is None:
            return self.params
        p = self.params
        fixed = p.embedding_params() + p.norm_params_per_layer() // 2
        budget = self.declared_params - fixed
        layers = max(1, round(budget / p.params_per_layer()))
        return p.model_copy(update={"layers": layers})

    def parameter_count(self) -> int:
        """Weight count of the model that is actually built (after preset scaling)."""
        return self.effective_params.parameter_count()

    def headline_parameter_count(self) -> float:
        """``declared_params`` when set, otherwise the derived count.

        This is the figure a preset is named for; the built model lands near it,
        not on it.
        """
        if self.declared_params is not None:
            return self.declared_params
        return float(self.parameter_count())

    def declared_vs_derived_error(self) -> float | None:
        """Relative gap between the declared headline and what layer scaling achieves.

        M3 emits an assumption line whenever this exceeds 1%.
        """
        if self.declared_params is None:
            return None
        derived = float(self.parameter_count())
        return abs(self.declared_params - derived) / derived


# -- CNN --------------------------------------------------------------------


class Activation(StrEnum):
    RELU = "relu"
    RELU6 = "relu6"
    HARDSWISH = "hardswish"
    SWISH = "swish"
    NONE = "none"


class PoolKind(StrEnum):
    MAX = "max"
    AVG = "avg"
    GLOBAL_AVG = "global_avg"


class InputSpec(SpecModel):
    """Input tensor shape for a CNN."""

    batch: int = Field(default=1, gt=0)
    channels: int = Field(gt=0)
    height: int = Field(gt=0)
    width: int = Field(gt=0)


class ConvLayer(SpecModel):
    """A convolution. ``groups == out_channels == in_channels`` is a depthwise conv."""

    type: Literal["conv"]
    name: str = Field(min_length=1)
    out_channels: int = Field(gt=0)
    kernel: tuple[int, int] = (3, 3)
    stride: int = Field(default=1, gt=0)
    padding: int = Field(default=0, ge=0)
    groups: int | None = Field(default=None, gt=0, description="None = dense; -1 sentinel unused")
    depthwise: bool = False
    bias: bool = False
    norm: NormType = NormType.BATCHNORM
    activation: Activation = Activation.RELU


class MBConvLayer(SpecModel):
    """An inverted-residual (MobileNet) block: 1x1 expand, depthwise KxK, 1x1 project."""

    type: Literal["mbconv"]
    name: str = Field(min_length=1)
    expand_channels: int = Field(gt=0)
    out_channels: int = Field(gt=0)
    kernel: tuple[int, int] = (3, 3)
    stride: int = Field(default=1, gt=0)
    squeeze_excite: bool = False
    activation: Activation = Activation.RELU


class PoolLayer(SpecModel):
    type: Literal["pool"]
    name: str = Field(min_length=1)
    kind: PoolKind
    kernel: tuple[int, int] | None = None
    stride: int = Field(default=1, gt=0)


class LinearLayer(SpecModel):
    type: Literal["linear"]
    name: str = Field(min_length=1)
    out_features: int = Field(gt=0)
    bias: bool = True
    activation: Activation = Activation.NONE


CNNLayer = Annotated[
    ConvLayer | MBConvLayer | PoolLayer | LinearLayer,
    Field(discriminator="type"),
]


class CNNSpec(_ModelBase):
    """An explicit-layer convolutional network."""

    family: Literal[ModelFamily.CNN]
    input: InputSpec
    layers: list[CNNLayer] = Field(min_length=1)

    def parameter_count(self) -> int:
        """Not derivable without channel propagation, which is the M2 graph builder's job.

        Raises rather than returning a wrong number: a CNN's parameter count
        depends on the input channel count of every layer, which only exists once
        the graph has been built.
        """
        raise NotImplementedError(
            f"model {self.id!r}: CNN parameter counts require channel propagation through the "
            f"graph; available from the M2 builder, not from the spec alone"
        )


# -- custom -----------------------------------------------------------------


class CustomOp(SpecModel):
    """A user-supplied operation with hand-specified cost. The escape hatch of §4.2."""

    name: str = Field(min_length=1)
    flops: Flops
    bytes: Bytes
    parallel_dims: dict[str, int] = Field(default_factory=dict)
    weight_bytes: float = Field(default=0.0, ge=0.0)


class CustomSpec(_ModelBase):
    """An explicit op list — models anything the parametric flavours cannot express."""

    family: Literal[ModelFamily.CUSTOM]
    ops: list[CustomOp] = Field(min_length=1)

    def parameter_count(self) -> int:
        """Inferred from declared ``weight_bytes``, assuming 1 byte per parameter.

        Only meaningful if the author populated ``weight_bytes``; returns 0 otherwise.
        """
        return math.floor(sum(op.weight_bytes for op in self.ops))


ModelSpec = Annotated[
    TransformerSpec | CNNSpec | CustomSpec,
    Field(discriminator="family"),
]
"""Any model profile. Discriminated on ``family`` so a bad value names the allowed set."""
