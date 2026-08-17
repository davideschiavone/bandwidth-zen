"""What each operation computes, written out so it can be checked by hand.

The engine reports that an operation costs 512 OP. This module says *why*: the
operand shapes, the algebra, the flop count as an arithmetic expression, and the
loop nest that would produce it. A reader can then reproduce the number with a
calculator, or write the same loops and compare — which is the only way to
believe a performance model that has never been calibrated.

**The counts are not recomputed here.** Each explanation carries the expression
(``2·M·N·K = 2·4·8·8``) and the value comes from :func:`bwz.operators.base.cost_of`,
so the text and the model cannot drift: :func:`check` asserts the printed
expression evaluates to the cost the engine used, and a test runs it over every
operation of every shipped profile.

Pure, like everything else outside ``api/`` and ``cli.py``: takes a graph, returns
strings.
"""

from __future__ import annotations

from dataclasses import dataclass

from bwz.calibration import RMSNORM_FLOPS_PER_ELEMENT, SOFTMAX_FLOPS_PER_SCORE
from bwz.graph.ops import (
    AttentionAttrs,
    ComputeGraph,
    ConvAttrs,
    ElementwiseAttrs,
    EmbeddingAttrs,
    MatmulAttrs,
    NormAttrs,
    Operation,
    PoolAttrs,
)
from bwz.operators.base import TensorTable, cost_of


@dataclass(frozen=True, slots=True)
class Explanation:
    """One operation, in four registers: shapes, algebra, arithmetic, code."""

    op_id: str
    op_type: str
    shapes: str
    """Operand extents, e.g. ``A[4,8] x B[8,8] -> C[4,8]``."""
    algebra: str
    """The mathematics in one line."""
    arithmetic: str
    """The flop count as an expression with its value, e.g. ``2·M·N·K = 2·4·8·8 = 512``."""
    flops: float
    code: str
    """A loop nest that performs exactly this arithmetic. Pseudo-C: indices and
    counts are real, types and memory layout are not."""


def explain_graph(graph: ComputeGraph) -> tuple[Explanation, ...]:
    """Every operation in the graph, in execution order."""
    return tuple(explain(op, graph.tensors) for op in graph.ops)


def explain(op: Operation, tensors: TensorTable) -> Explanation:
    """Describe one operation. Falls back to a shape-only note for families that
    have no closed form worth writing out."""
    cost = cost_of(op, tensors)
    flops = cost.flops
    attrs = op.attrs

    if isinstance(attrs, MatmulAttrs):
        return _matmul(op, attrs, flops)
    if isinstance(attrs, AttentionAttrs):
        return _attention(op, attrs, flops)
    if isinstance(attrs, NormAttrs):
        return _norm(op, attrs, flops)
    if isinstance(attrs, ElementwiseAttrs):
        return _elementwise(op, attrs, flops)
    if isinstance(attrs, EmbeddingAttrs):
        return _embedding(op, attrs, flops, cost.weight_bytes + cost.output_bytes)
    if isinstance(attrs, ConvAttrs):
        return _conv(op, attrs, flops)
    if isinstance(attrs, PoolAttrs):
        return _pool(op, attrs, flops)
    return Explanation(
        op.id, op.op_type.value, "—", "—", f"{flops:,.0f} OP as declared", flops, "/* declared */"
    )


def _matmul(op: Operation, a: MatmulAttrs, flops: float) -> Explanation:
    return Explanation(
        op_id=op.id,
        op_type=op.op_type.value,
        shapes=f"A[{a.m},{a.k}] x B[{a.k},{a.n}] -> C[{a.m},{a.n}]",
        algebra="C[m,n] = Σ_k A[m,k] · B[k,n]",
        arithmetic=f"2·M·N·K = 2·{a.m}·{a.n}·{a.k} = {flops:,.0f}",
        flops=flops,
        code=f"""for (m = 0; m < {a.m}; ++m)
  for (n = 0; n < {a.n}; ++n) {{
    acc = 0;
    for (k = 0; k < {a.k}; ++k)
      acc += A[m][k] * B[k][n];   /* 1 multiply + 1 add = 2 OP */
    C[m][n] = acc;
  }}""",
    )


def _attention(op: Operation, a: AttentionAttrs, flops: float) -> Explanation:
    positions = (
        a.q_len * (a.kv_len - a.q_len) + a.q_len * (a.q_len + 1) // 2
        if a.causal and a.q_len > 1
        else a.q_len * a.kv_len
    )
    scored = a.batch * a.heads * positions
    # No nested /* */: C does not allow them, and this is meant to compile if
    # someone actually pastes it.
    mask = "k <= q, causal" if a.causal and a.q_len > 1 else "every k, bidirectional"
    softmax = int(SOFTMAX_FLOPS_PER_SCORE)
    return Explanation(
        op_id=op.id,
        op_type=op.op_type.value,
        shapes=(
            f"Q[{a.batch},{a.heads},{a.q_len},{a.head_dim}] · "
            f"K[{a.batch},{a.kv_heads},{a.kv_len},{a.head_dim}]ᵀ -> "
            f"S[{a.batch},{a.heads},{a.q_len},{a.kv_len}] · V -> O"
        ),
        algebra="O = softmax(Q·Kᵀ / √d) · V",
        arithmetic=(
            f"scored = batch·heads·positions = {a.batch}·{a.heads}·{positions} = {scored:,}; "
            f"2·2·scored·d + {softmax}·scored = "
            f"{2 * 2 * scored * a.head_dim:,} + {softmax * scored:,} = {flops:,.0f}"
        ),
        flops=flops,
        code=f"""for (h = 0; h < {a.heads}; ++h) {{
  for (q = 0; q < {a.q_len}; ++q) {{
    for (k = 0; k < {a.kv_len}; ++k) {{        /* {mask} */
      s = 0;
      for (d = 0; d < {a.head_dim}; ++d)
        s += Q[h][q][d] * K[h % {a.kv_heads}][k][d];   /* 2 OP */
      S[h][q][k] = s / sqrt({a.head_dim});
    }}
    softmax(S[h][q], {a.kv_len});              /* {softmax} OP per score */
    for (d = 0; d < {a.head_dim}; ++d) {{
      o = 0;
      for (k = 0; k < {a.kv_len}; ++k)
        o += S[h][q][k] * V[h % {a.kv_heads}][k][d];   /* 2 OP */
      O[h][q][d] = o;
    }}
  }}
}}""",
    )


def _norm(op: Operation, a: NormAttrs, flops: float) -> Explanation:
    return Explanation(
        op_id=op.id,
        op_type=op.op_type.value,
        shapes=f"X[{a.rows},{a.width}] -> Y[{a.rows},{a.width}]",
        algebra=(
            "y = x / rms(x) · g"
            if a.flops_per_element == RMSNORM_FLOPS_PER_ELEMENT
            else "y = (x - mean) / stddev · g + b"
        ),
        arithmetic=(
            f"rows·width·{a.flops_per_element:g} = "
            f"{a.rows}·{a.width}·{a.flops_per_element:g} = {flops:,.0f}"
        ),
        flops=flops,
        code=f"""for (r = 0; r < {a.rows}; ++r) {{
  acc = 0;
  for (i = 0; i < {a.width}; ++i) acc += X[r][i] * X[r][i];
  scale = rsqrt(acc / {a.width} + eps);
  for (i = 0; i < {a.width}; ++i) Y[r][i] = X[r][i] * scale * g[i];
}}                                    /* {a.flops_per_element:g} OP per element */""",
    )


def _elementwise(op: Operation, a: ElementwiseAttrs, flops: float) -> Explanation:
    return Explanation(
        op_id=op.id,
        op_type=op.op_type.value,
        shapes=f"{a.n_inputs} x X[{a.elements:,}] -> Y[{a.elements:,}]",
        algebra="y = f(x…) elementwise",
        arithmetic=(
            f"elements·{a.flops_per_element:g} = "
            f"{a.elements:,}·{a.flops_per_element:g} = {flops:,.0f}"
        ),
        flops=flops,
        code=f"""for (i = 0; i < {a.elements}; ++i)
  Y[i] = f(X0[i]{", X1[i]" if a.n_inputs > 1 else ""});
                                    /* {a.flops_per_element:g} OP per element */""",
    )


def _embedding(op: Operation, a: EmbeddingAttrs, flops: float, moved: float) -> Explanation:
    return Explanation(
        op_id=op.id,
        op_type=op.op_type.value,
        shapes=f"ids[{a.tokens}] -> Y[{a.tokens},{a.width}]",
        algebra="y[t,:] = table[ids[t], :]",
        # Zero arithmetic is not zero cost, and a panel that says only "0 OP"
        # reads as free. A copy transforms nothing — that is why it contributes
        # no FLOPs, and why the standard 2·N·D transformer count excludes it —
        # but it moves bytes and pays a dispatch like anything else.
        arithmetic=(
            f"a gather: no arithmetic, {flops:,.0f} OP. The cost is movement — "
            f"{a.tokens} rows of {a.width}, {moved:,.0f} bytes — plus one dispatch"
        ),
        flops=flops,
        code=f"""for (t = 0; t < {a.tokens}; ++t)
  memcpy(Y[t], table[ids[t]], {a.width} * sizeof(elem));
      /* a move, not an operation: 0 FLOP, {moved:,.0f} bytes, 1 dispatch.
         The table is footprint; only the rows touched are traffic. */""",
    )


def _conv(op: Operation, a: ConvAttrs, flops: float) -> Explanation:
    per_group = a.in_channels // a.groups
    return Explanation(
        op_id=op.id,
        op_type=op.op_type.value,
        shapes=(
            f"X[{a.batch},{a.in_channels},{a.in_height},{a.in_width}] * "
            f"W[{a.out_channels},{per_group},{a.kernel_h},{a.kernel_w}] -> "
            f"Y[{a.batch},{a.out_channels},{a.out_height},{a.out_width}]"
        ),
        algebra="y[o,i,j] = Σ_c Σ_u Σ_v x[c, i·s+u, j·s+v] · w[o,c,u,v]",
        arithmetic=(
            f"2·N·Co·H·W·(Ci/g)·Kh·Kw = 2·{a.batch}·{a.out_channels}·{a.out_height}·"
            f"{a.out_width}·{per_group}·{a.kernel_h}·{a.kernel_w} = {flops:,.0f}"
        ),
        flops=flops,
        code=f"""for (o = 0; o < {a.out_channels}; ++o)
  for (i = 0; i < {a.out_height}; ++i)
    for (j = 0; j < {a.out_width}; ++j) {{
      acc = 0;
      for (c = 0; c < {per_group}; ++c)
        for (u = 0; u < {a.kernel_h}; ++u)
          for (v = 0; v < {a.kernel_w}; ++v)
            acc += X[c][i*s+u][j*s+v] * W[o][c][u][v];
      Y[o][i][j] = acc;
    }}""",
    )


def _pool(op: Operation, a: PoolAttrs, flops: float) -> Explanation:
    return Explanation(
        op_id=op.id,
        op_type=op.op_type.value,
        shapes=f"X[...] -> Y[{a.batch},{a.channels},{a.out_height},{a.out_width}]",
        algebra=f"y = reduce over each {a.kernel_h}x{a.kernel_w} window",
        arithmetic=f"windows · elements = {flops:,.0f}",
        flops=flops,
        code=f"""for (c = 0; c < {a.channels}; ++c)
  for (i = 0; i < {a.out_height}; ++i)
    for (j = 0; j < {a.out_width}; ++j)
      Y[c][i][j] = reduce(X[c], i, j, {a.kernel_h}, {a.kernel_w});""",
    )


def check(graph: ComputeGraph) -> tuple[str, ...]:
    """Explanations whose printed expression disagrees with the engine's count.

    The text is written by hand and the number is not, so this is the guard
    against the two drifting apart. Empty is the passing case.
    """
    bad: list[str] = []
    for op, explanation in zip(graph.ops, explain_graph(graph), strict=True):
        expected = cost_of(op, graph.tensors).flops
        if abs(explanation.flops - expected) > 1e-6 * max(1.0, expected):
            bad.append(f"{op.id}: explained {explanation.flops} against {expected}")
        if f"{expected:,.0f}" not in explanation.arithmetic:
            bad.append(f"{op.id}: expression does not print {expected:,.0f}")
    return tuple(bad)
