"""Ad-hoc kernel probes: a spec built straight from shape arguments, no profile.

``docs/CORRECTIONS.md`` D39. Two kernels, ``matmul`` and ``encoder-layer`` — the
single place ``bwz matmul``/``bwz encoder-layer`` and their ``--timeline``/``--animate``
``--matmul``/``--encoder`` build the probe, so the report and the figure for the
same probe cannot silently describe two different specs the way they had drifted
to before this module existed: a placeholder id (``"p"``) on the plotted matmul, no
per-operand widths on the plotted matmul at all, and the plotted encoder silently
hardcoding three architecture knobs the reported one exposes as flags.

A model loaded from a profile (``bwz run -m <id>``) is a different thing —
possibly many kernels, many layers, real weights. These two build exactly one
kernel invocation each, which is what "kernel" means here: the smallest unit the
engine costs, probed directly rather than composed into a model.
"""

from __future__ import annotations

from bwz.spec import DType, FFNType, MatmulSpec, NormType, TransformerSpec


def matmul_kernel(
    m: int,
    n: int,
    k: int,
    *,
    a_dtype: DType,
    b_dtype: DType,
    out_dtype: DType | None = None,
) -> MatmulSpec:
    """One ``A[M,K] x B[K,N] -> C[M,N]``, id and name derived from the shape.

    Only the shape enters the id, not dtype: two matmuls of the same M, N, K at
    different precision are still "the same probe" read at a different width.
    """
    return MatmulSpec.model_validate(
        {
            "id": f"matmul_{m}x{n}x{k}",
            "name": f"matmul {m}x{n}x{k}",
            "family": "matmul",
            "m": m,
            "n": n,
            "k": k,
            "a_dtype": a_dtype,
            "b_dtype": b_dtype,
            "out_dtype": out_dtype,
        }
    )


def encoder_layer_kernel(
    *,
    dmodel: int,
    nheads: int,
    ffn: int,
    vocab: int,
    tokens: int,
    ffn_type: FFNType = FFNType.RELU,
    norm: NormType = NormType.RMSNORM,
    tie_embeddings: bool = True,
) -> TransformerSpec:
    """One encoder layer, id and name derived from the shape.

    The transformer counterpart of :func:`matmul_kernel`. ``ffn_type``/``norm``/
    ``tie_embeddings`` default to the plainest configuration the spec allows —
    the same defaults ``bwz encoder-layer`` itself uses — so a caller that only
    knows the shape gets the same encoder either way. Like :func:`matmul_kernel`,
    the id encodes shape only, not these three architecture knobs.

    Unlike :class:`bwz.spec.model_spec.TransformerParams` (which a loaded
    ``--model`` profile uses and which lets ``head_dim`` be set independently
    of ``dmodel``/``nheads`` — real GQA profiles like Gemma-3 need that, D44),
    this probe has no ``head_dim`` knob at all: it is always ``dmodel //
    nheads``, and a shape that does not divide evenly is rejected outright
    rather than silently floored. One shape typed on the command line should
    have one unambiguous meaning.
    """
    if dmodel % nheads != 0:
        raise ValueError(
            f"dmodel ({dmodel}) is not divisible by nheads ({nheads}); choose a head count "
            f"dividing {dmodel} evenly"
        )
    return TransformerSpec.model_validate(
        {
            "id": f"encoder_layer_d{dmodel}_h{nheads}_ffn{ffn}_s{tokens}",
            "name": f"1-layer encoder d={dmodel} h={nheads} ffn={ffn}",
            "family": "transformer_encoder",
            "hypothetical": True,
            "params": {
                "layers": 1,
                "hidden": dmodel,
                "heads": nheads,
                "head_dim": None,
                "ffn_hidden": ffn,
                "ffn_type": ffn_type,
                "vocab": vocab,
                "max_context": max(tokens, 1),
                "norm": norm,
                "positional": "none",
                "tie_embeddings": tie_embeddings,
            },
        }
    )
