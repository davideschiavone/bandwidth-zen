"""Expand a :class:`TransformerSpec` into a prefill graph or a decode graph.

**Two graphs, never one with a flag** (CLAUDE.md #6). The difference is a single
number — the row count ``M`` of every projection — but that number moves each
GEMM by three orders of magnitude in arithmetic intensity, and with it the
bottleneck, the utilisation and the right hardware. Prefill has ``M = batch * S``
and reads each weight once to do ``S`` tokens of work; decode has ``M = batch``
and reads the same weights to do one token of work.

``docs/MODEL.md`` §4.
"""

from __future__ import annotations

from bwz.calibration import (
    GELU_FLOPS_PER_ELEMENT,
    LAYERNORM_FLOPS_PER_ELEMENT,
    RMSNORM_FLOPS_PER_ELEMENT,
    ROPE_FLOPS_PER_ELEMENT,
    SILU_FLOPS_PER_ELEMENT,
)
from bwz.graph.ops import (
    AttentionAttrs,
    ComputeGraph,
    ElementwiseAttrs,
    EmbeddingAttrs,
    GraphPhase,
    MatmulAttrs,
    NormAttrs,
    Operation,
    OpType,
    Tensor,
    TensorKind,
)
from bwz.spec.deployment import AttentionImpl, DeploymentSpec
from bwz.spec.dtypes import DType
from bwz.spec.model_spec import FFNType, NormType, PositionalType, TransformerSpec

_NORM_FLOPS = {
    NormType.RMSNORM: RMSNORM_FLOPS_PER_ELEMENT,
    NormType.LAYERNORM: LAYERNORM_FLOPS_PER_ELEMENT,
    NormType.BATCHNORM: LAYERNORM_FLOPS_PER_ELEMENT,
    NormType.NONE: 0.0,
}

_FFN_ACTIVATION_FLOPS = {
    FFNType.SWIGLU: SILU_FLOPS_PER_ELEMENT + 1.0,  # SiLU on the gate, then the gating multiply
    FFNType.GEGLU: GELU_FLOPS_PER_ELEMENT + 1.0,
    FFNType.GELU: GELU_FLOPS_PER_ELEMENT,
    FFNType.RELU: 1.0,
}


class _Builder:
    """Accumulates tensors and operations while walking the model."""

    def __init__(self, model: TransformerSpec, deployment: DeploymentSpec, phase: GraphPhase):
        self.model = model
        self.params = model.effective_params
        """Post-preset-scaling shape: what actually gets built."""
        self.deployment = deployment
        self.phase = phase
        self.tensors: dict[str, Tensor] = {}
        self.ops: list[Operation] = []

        self.w_dtype = deployment.precision.weights
        self.a_dtype = deployment.precision.activations
        self.kv_dtype = deployment.precision.kv_cache

        self.batch = deployment.batch
        if phase is GraphPhase.PREFILL:
            self.seq = deployment.input_tokens
            self.kv_len = deployment.input_tokens
        else:
            self.seq = 1
            self.kv_len = deployment.context_tokens
        self.rows = self.batch * self.seq
        """``M`` for every projection: batch x sequence at prefill, batch at decode."""

    # -- helpers ------------------------------------------------------------

    def weight(self, name: str, shape: tuple[int, ...]) -> str:
        self.tensors.setdefault(name, Tensor(name, shape, self.w_dtype, TensorKind.WEIGHT))
        return name

    def activation(self, name: str, shape: tuple[int, ...]) -> str:
        self.tensors.setdefault(name, Tensor(name, shape, self.a_dtype, TensorKind.ACTIVATION))
        return name

    def kv(self, name: str, shape: tuple[int, ...]) -> str:
        self.tensors.setdefault(name, Tensor(name, shape, self.kv_dtype, TensorKind.KV_CACHE))
        return name

    def emit(self, op: Operation) -> str:
        self.ops.append(op)
        return op.outputs[0]

    def matmul(self, op_id: str, source: str, weight_name: str, n: int, layer: int | None) -> str:
        """``[rows, k] x [k, n]``, with ``k`` taken from the weight's first axis."""
        k = self.tensors[weight_name].shape[0]
        out = self.activation(f"{op_id}.out", (self.rows, n))
        return self.emit(
            Operation(
                id=op_id,
                op_type=OpType.MATMUL,
                attrs=MatmulAttrs(m=self.rows, n=n, k=k),
                inputs=(source,),
                weights=(weight_name,),
                outputs=(out,),
                layer=layer,
            )
        )

    def norm(self, op_id: str, source: str, layer: int | None) -> str:
        """RMSNorm carries a scale per channel; LayerNorm carries a scale *and* a bias.

        Sizing every norm as ``(hidden,)`` would undercount GPT-3 by 2.37 M
        parameters across its 193 LayerNorms — small in relative terms, but it
        breaks the graph-versus-spec cross-check, which is precisely the test that
        exists to catch it.
        """
        hidden = self.params.hidden
        if self.params.norm is NormType.NONE:
            return source
        rows_per_norm = 1 if self.params.norm is NormType.RMSNORM else 2
        weight_name = self.weight(f"{op_id}.scale", (rows_per_norm, hidden))
        out = self.activation(f"{op_id}.out", (self.rows, hidden))
        return self.emit(
            Operation(
                id=op_id,
                op_type=OpType.NORM,
                attrs=NormAttrs(
                    rows=self.rows, width=hidden, flops_per_element=_NORM_FLOPS[self.params.norm]
                ),
                inputs=(source,),
                weights=(weight_name,),
                outputs=(out,),
                layer=layer,
            )
        )

    def elementwise(
        self,
        op_id: str,
        sources: tuple[str, ...],
        shape: tuple[int, ...],
        flops_per_element: float,
        layer: int | None,
    ) -> str:
        elements = 1
        for dim in shape:
            elements *= dim
        out = self.activation(f"{op_id}.out", shape)
        return self.emit(
            Operation(
                id=op_id,
                op_type=OpType.ELEMENTWISE,
                attrs=ElementwiseAttrs(
                    elements=elements,
                    n_inputs=len(sources),
                    flops_per_element=flops_per_element,
                ),
                inputs=sources,
                outputs=(out,),
                layer=layer,
            )
        )

    # -- the model ----------------------------------------------------------

    def build(self) -> ComputeGraph:
        p = self.params
        embedding = self.weight("embed_tokens", (p.vocab, p.hidden))
        hidden_state = self.activation("embed.out", (self.rows, p.hidden))
        self.ops.append(
            Operation(
                id="embed",
                op_type=OpType.EMBEDDING,
                attrs=EmbeddingAttrs(tokens=self.rows, width=p.hidden),
                weights=(embedding,),
                outputs=(hidden_state,),
            )
        )

        for layer in range(p.layers):
            hidden_state = self.block(hidden_state, layer)

        hidden_state = self.norm("final_norm", hidden_state, layer=None)
        self.lm_head(hidden_state, embedding)

        return ComputeGraph(
            name=f"{self.model.id}.{self.phase.value}",
            phase=self.phase,
            ops=tuple(self.ops),
            tensors=self.tensors,
        )

    def block(self, hidden_state: str, layer: int) -> str:
        p = self.params
        prefix = f"layer{layer}"

        normed = self.norm(f"{prefix}.attn_norm", hidden_state, layer)

        q = self.matmul(
            f"{prefix}.q_proj",
            normed,
            self.weight(f"{prefix}.q_proj.w", (p.hidden, p.q_width)),
            p.q_width,
            layer,
        )
        k = self.matmul(
            f"{prefix}.k_proj",
            normed,
            self.weight(f"{prefix}.k_proj.w", (p.hidden, p.kv_width)),
            p.kv_width,
            layer,
        )
        v = self.matmul(
            f"{prefix}.v_proj",
            normed,
            self.weight(f"{prefix}.v_proj.w", (p.hidden, p.kv_width)),
            p.kv_width,
            layer,
        )

        if p.positional is PositionalType.ROPE:
            q = self.elementwise(
                f"{prefix}.rope",
                (q, k),
                (self.rows, p.q_width + p.kv_width),
                ROPE_FLOPS_PER_ELEMENT,
                layer,
            )

        attn_out = self.attention(prefix, q, k, v, layer)
        projected = self.matmul(
            f"{prefix}.o_proj",
            attn_out,
            self.weight(f"{prefix}.o_proj.w", (p.q_width, p.hidden)),
            p.hidden,
            layer,
        )
        hidden_state = self.elementwise(
            f"{prefix}.attn_residual", (hidden_state, projected), (self.rows, p.hidden), 1.0, layer
        )

        normed = self.norm(f"{prefix}.ffn_norm", hidden_state, layer)
        activated = self.ffn(prefix, normed, layer)
        down = self.matmul(
            f"{prefix}.ffn_down",
            activated,
            self.weight(f"{prefix}.ffn_down.w", (p.ffn_hidden, p.hidden)),
            p.hidden,
            layer,
        )
        return self.elementwise(
            f"{prefix}.ffn_residual", (hidden_state, down), (self.rows, p.hidden), 1.0, layer
        )

    def attention(self, prefix: str, q: str, k: str, v: str, layer: int) -> str:
        """Attach the attention op, wiring KV traffic differently per phase.

        At prefill the K/V just computed *are* the cache being built, so the
        projections write it and attention reads what it wrote. At decode the
        cache already holds ``kv_len`` tokens that attention must stream, and the
        one new K/V row is rounding error next to it — this asymmetry is the
        reason decode's cost grows with context and prefill's does not.
        """
        p = self.params
        kv_shape = (self.batch, self.kv_len, p.kv_width)
        k_cache = self.kv(f"{prefix}.k_cache", kv_shape)
        v_cache = self.kv(f"{prefix}.v_cache", kv_shape)

        reads: tuple[str, ...]
        if self.phase is GraphPhase.PREFILL:
            reads = (q, k, v, k_cache, v_cache)
        else:
            reads = (q, k_cache, v_cache)

        materialize = self.deployment.attention_impl is AttentionImpl.VANILLA
        out = self.activation(f"{prefix}.attn.out", (self.rows, p.q_width))
        return self.emit(
            Operation(
                id=f"{prefix}.attn",
                op_type=OpType.ATTENTION,
                attrs=AttentionAttrs(
                    batch=self.batch,
                    heads=p.heads,
                    kv_heads=p.effective_kv_heads,
                    head_dim=p.effective_head_dim,
                    q_len=self.seq,
                    kv_len=self.kv_len,
                    causal=True,
                    materialize_scores=materialize,
                ),
                inputs=reads,
                outputs=(out,),
                layer=layer,
            )
        )

    def ffn(self, prefix: str, normed: str, layer: int) -> str:
        p = self.params
        up = self.matmul(
            f"{prefix}.ffn_up",
            normed,
            self.weight(f"{prefix}.ffn_up.w", (p.hidden, p.ffn_hidden)),
            p.ffn_hidden,
            layer,
        )
        sources: tuple[str, ...] = (up,)
        if p.ffn_type.is_gated:
            gate = self.matmul(
                f"{prefix}.ffn_gate",
                normed,
                self.weight(f"{prefix}.ffn_gate.w", (p.hidden, p.ffn_hidden)),
                p.ffn_hidden,
                layer,
            )
            sources = (gate, up)
        return self.elementwise(
            f"{prefix}.ffn_act",
            sources,
            (self.rows, p.ffn_hidden),
            _FFN_ACTIVATION_FLOPS[p.ffn_type],
            layer,
        )

    def lm_head(self, hidden_state: str, embedding: str) -> None:
        """Logits for the **last token only**, in both phases.

        Inference needs one distribution per generated token, not one per prompt
        token, and every serving stack slices before the projection. Computing all
        ``S`` would add ``2 * S * hidden * vocab`` FLOPs — for Gemma-3-4B at
        S=2048 that is 2.7 TFLOP against a 13.7 TFLOP forward pass, a 20% error.
        Recorded as an assumption where it is consumed.

        With tied embeddings the head reuses the embedding tensor, so the weight
        is counted once in the footprint.
        """
        p = self.params
        weight_name = (
            embedding if p.tie_embeddings else self.weight("lm_head.w", (p.hidden, p.vocab))
        )
        out = self.activation("lm_head.out", (self.batch, p.vocab))
        self.ops.append(
            Operation(
                id="lm_head",
                op_type=OpType.MATMUL,
                attrs=MatmulAttrs(m=self.batch, n=p.vocab, k=p.hidden),
                inputs=(hidden_state,),
                weights=(weight_name,),
                outputs=(out,),
            )
        )


def build_transformer_graph(
    model: TransformerSpec, deployment: DeploymentSpec, phase: GraphPhase
) -> ComputeGraph:
    """Expand *model* into a graph for *phase*.

    Raises on :attr:`GraphPhase.STATIC`, which is the CNN phase — a transformer
    always has a prefill/decode distinction.
    """
    if phase is GraphPhase.STATIC:
        raise ValueError(
            f"model {model.id!r} is a transformer; build it for phase 'prefill' or 'decode', "
            f"not 'static' (CLAUDE.md #6: prefill and decode are different machines)"
        )
    return _Builder(model, deployment, phase).build()


def weight_dtype_bytes(deployment: DeploymentSpec) -> float:
    """Bytes per weight element — the ``w_bytes`` of ``W = params x w_bytes``."""
    from bwz.spec.dtypes import bytes_per_element

    return bytes_per_element(deployment.precision.weights)


__all__ = ["DType", "build_transformer_graph", "weight_dtype_bytes"]
