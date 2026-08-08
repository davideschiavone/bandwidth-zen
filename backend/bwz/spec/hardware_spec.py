"""Chip profiles: compute units, memory levels, interconnect.

Schema reference: ``docs/SCHEMA.md`` §1; source schema: ``PROMPT.md`` §4.1.

**The v1 machine model (docs/CORRECTIONS.md D5a)** is three elements and no more:

===================  ==========================================================
External DRAM/HBM    the *only* bandwidth ceiling
On-chip SRAM         **capacity only** — sets the resident weight fraction and
                     the headroom that makes double buffering possible
Compute engine       the TOPS ceiling
===================  ==========================================================

Every memory level still carries ``bandwidth_bytes_per_s`` because PROMPT.md
§4.1 specifies it and the M8 hierarchical roofline will need it, but v1 analysis
reads bandwidth only from the deepest level and capacity only from the
shallowest. Loading a profile records that in ``report.assumptions``.

MAC-to-OP doubling happens in :meth:`ComputeUnit.peak_flops_per_s` and nowhere
else in the codebase (CLAUDE.md #5).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from bwz.spec.base import Identifier, SourceUrl, SpecModel
from bwz.spec.dtypes import DType
from bwz.spec.quantities import Bytes, BytesPerSecond, Fraction, Seconds, Watts


class Dataflow(StrEnum):
    """Which operand stays resident in the PE array.

    v1 honours the profile's declaration; the loop-order search that would
    *derive* it is deferred to M8 (docs/CORRECTIONS.md D5).
    """

    WEIGHT_STATIONARY = "ws"
    OUTPUT_STATIONARY = "os"
    ROW_STATIONARY = "rs"


class Topology(StrEnum):
    """Interconnect topology, consumed by the M5 collective cost model."""

    FULLY_CONNECTED = "fully_connected"
    RING = "ring"
    MESH = "mesh"
    FAT_TREE = "fat_tree"
    SWITCHED = "switched"


class ComputeUnit(SpecModel):
    """One class of compute engine on the chip (tensor core, CUDA core, NPU core).

    ``ops_per_cycle_per_unit`` is **MACs** per cycle per unit, per CLAUDE.md #5.
    A 512x512 systolic array contributes 262144.
    """

    name: str = Field(min_length=1)
    count: int = Field(gt=0)
    ops_per_cycle_per_unit: float = Field(gt=0, description="MACs per cycle per unit")
    supported_dtypes: list[DType] = Field(min_length=1)
    dtype_multipliers: dict[DType, float] = Field(default_factory=dict)
    structured_sparsity_speedup: float = Field(default=1.0, ge=1.0)
    systolic_dims: tuple[int, int] | None = Field(
        default=None, description="Rows x columns of the PE array; drives the M3 tail-effect model"
    )
    dataflow: Dataflow = Dataflow.WEIGHT_STATIONARY

    @model_validator(mode="after")
    def _check_multipliers(self) -> ComputeUnit:
        unknown = sorted(d.value for d in self.dtype_multipliers if d not in self.supported_dtypes)
        if unknown:
            supported = sorted(d.value for d in self.supported_dtypes)
            raise ValueError(
                f"compute unit {self.name!r}: dtype_multipliers names {unknown} which are not in "
                f"supported_dtypes {supported}; add them to supported_dtypes or remove the "
                f"multiplier"
            )
        bad = sorted(f"{d.value}={m}" for d, m in self.dtype_multipliers.items() if m <= 0)
        if bad:
            raise ValueError(
                f"compute unit {self.name!r}: dtype_multipliers must be > 0, got {bad}"
            )
        if self.systolic_dims is not None and min(self.systolic_dims) < 1:
            raise ValueError(
                f"compute unit {self.name!r}: systolic_dims must be positive, got "
                f"{self.systolic_dims}"
            )
        return self

    def supports(self, dtype: DType) -> bool:
        """True if this unit can execute *dtype* at all."""
        return dtype in self.supported_dtypes

    def peak_flops_per_s(self, clock_hz: float, dtype: DType, *, sparsity: bool = False) -> float:
        """Peak throughput in OP/s: ``count x MACs/cycle x clock x multiplier x 2``.

        **This is the only place in the codebase where MACs are doubled into
        operations** (CLAUDE.md #5). The name says FLOP for consistency with the
        rest of the engine; for integer dtypes the unit is OP/s.

        Worked example (H100 SXM5 fp16 dense, docs/MODEL.md):
        ``528 x 512 x 1.830e9 x 1.0 x 2 = 9.894e14`` = 989.4 TFLOP/s.

        Worked example (a 512x512 bit-serial INT8 NPU core at 0.8 GHz, whose
        ``int8`` multiplier of 0.125 encodes the 8-cycles-per-INT8-MAC tax):
        ``1 x 262144 x 0.8e9 x 0.125 x 2 = 5.24e13`` = 52.4 TOPS per core.

        Returns 0.0 if the unit does not support *dtype*, so that callers can sum
        or max over a heterogeneous unit list without filtering first.
        """
        if not self.supports(dtype):
            return 0.0
        multiplier = self.dtype_multipliers.get(dtype, 1.0)
        speedup = self.structured_sparsity_speedup if sparsity else 1.0
        macs_per_s = self.count * self.ops_per_cycle_per_unit * clock_hz * multiplier * speedup
        return 2.0 * macs_per_s


class MemoryLevel(SpecModel):
    """One level of the memory hierarchy. ``level`` 1 is closest to the PEs."""

    name: str = Field(min_length=1)
    level: int = Field(gt=0)
    capacity_bytes: Bytes
    bandwidth_bytes_per_s: BytesPerSecond
    latency_ns: float = Field(default=0.0, ge=0.0)


class Interconnect(SpecModel):
    """A chip-to-chip link. Unused until M5; validated here so profiles are complete."""

    name: str = Field(min_length=1)
    bandwidth_bytes_per_s: BytesPerSecond
    latency_s: Seconds
    topology: Topology


class InterconnectSet(SpecModel):
    """Intra-node (e.g. NVLink) and inter-node (e.g. InfiniBand) links."""

    intra_node: Interconnect | None = None
    inter_node: Interconnect | None = None


class HardwareSpec(SpecModel):
    """A chip profile.

    Provenance rules (docs/CORRECTIONS.md D7): a profile describing a real
    product must carry ``source_url``. A profile that is *not* a product declares
    ``hypothetical: true`` and points at what it was derived from. Individual
    fields that are engineering estimates rather than published figures are named
    in ``estimates``, and every estimate the analysis actually touches is
    propagated into ``report.assumptions``.
    """

    id: Identifier
    name: str = Field(min_length=1)
    vendor: str = Field(min_length=1)
    source_url: SourceUrl | None = None
    hypothetical: bool = False
    derived_from: Identifier | None = None
    estimates: dict[str, str] = Field(
        default_factory=dict,
        description="field name -> why this value is an estimate rather than a published figure",
    )

    process_nm: float | None = Field(default=None, gt=0)
    clock_ghz: float = Field(gt=0)
    compute_units: list[ComputeUnit] = Field(min_length=1)
    memory: list[MemoryLevel] = Field(min_length=1)

    usable_memory_fraction: Fraction | None = None
    async_copy_engines: int = Field(default=1, ge=0)
    kernel_launch_overhead_s: Seconds | None = None
    tdp_w: Watts | None = None
    static_power_w: Watts | None = None
    interconnect: InterconnectSet | None = None
    cost_usd: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_provenance(self) -> HardwareSpec:
        if not self.hypothetical and self.source_url is None:
            raise ValueError(
                f"chip {self.id!r}: source_url is required for a profile describing a real "
                f"product (CLAUDE.md: never commit profile YAML without a source_url). If this "
                f"chip is not a product, set 'hypothetical: true' and 'derived_from: <chip_id>'."
            )
        if self.derived_from is not None and not self.hypothetical:
            raise ValueError(
                f"chip {self.id!r}: derived_from is only meaningful on a hypothetical profile; "
                f"set 'hypothetical: true' or remove derived_from"
            )
        known = set(type(self).model_fields)
        unknown = sorted({k.split(".")[0] for k in self.estimates} - known)
        if unknown:
            raise ValueError(
                f"chip {self.id!r}: estimates names unknown field(s) {unknown}; allowed roots are "
                f"{sorted(known)}"
            )
        return self

    @model_validator(mode="after")
    def _check_memory_levels(self) -> HardwareSpec:
        levels = [m.level for m in self.memory]
        if len(set(levels)) != len(levels):
            raise ValueError(
                f"chip {self.id!r}: duplicate memory level(s) in {levels}; each level must appear "
                f"once, numbered from 1 (closest to the PEs) outwards"
            )
        if levels != sorted(levels):
            raise ValueError(
                f"chip {self.id!r}: memory levels must be listed innermost-first, got {levels}; "
                f"reorder so level numbers ascend"
            )
        return self

    # -- derived quantities -------------------------------------------------

    @property
    def clock_hz(self) -> float:
        """Clock in SI hertz. ``clock_ghz`` is the human-facing YAML field."""
        return self.clock_ghz * 1e9

    @property
    def dram(self) -> MemoryLevel:
        """The deepest memory level: the only bandwidth ceiling in v1 (D5a)."""
        return self.memory[-1]

    @property
    def on_chip(self) -> MemoryLevel:
        """The shallowest memory level: the tile buffer. v1 reads its *capacity* only (D5a)."""
        return self.memory[0]

    @property
    def on_chip_capacity_bytes(self) -> float:
        """Total on-chip capacity available for weight residency and double buffering.

        Sums every level above DRAM, since a single-level chip has ``on_chip is
        dram`` and must report zero rather than counting DRAM twice.
        """
        if len(self.memory) == 1:
            return 0.0
        return sum(m.capacity_bytes for m in self.memory[:-1])

    def peak_flops_per_s(self, dtype: DType, *, sparsity: bool = False) -> float:
        """Peak chip throughput for *dtype*, in OP/s.

        Takes the **maximum** over compute units rather than the sum: a GEMM runs
        on the tensor cores or the vector cores, not both at once, and vendor
        headline figures quote the dominant unit. Recorded as an assumption at
        M3. Raises if no unit supports *dtype* — an actionable error beats a
        silent zero that later divides.
        """
        if not self.supports(dtype):
            available = sorted({d.value for u in self.compute_units for d in u.supported_dtypes})
            raise ValueError(
                f"chip {self.id!r} ({self.name}) has no compute unit supporting dtype "
                f"{dtype.value!r}; supported dtypes are {available}"
            )
        return max(
            u.peak_flops_per_s(self.clock_hz, dtype, sparsity=sparsity) for u in self.compute_units
        )

    def supports(self, dtype: DType) -> bool:
        """True if any compute unit can execute *dtype*."""
        return any(u.supports(dtype) for u in self.compute_units)

    def ridge_point_flops_per_byte(self, dtype: DType, *, sparsity: bool = False) -> float:
        """Arithmetic intensity at which the DRAM roofline meets the compute roofline.

        ``peak_flops_per_s / dram.bandwidth_bytes_per_s`` (docs/MODEL.md). An
        operation with intensity below this is DRAM-bound, above it is
        compute-bound.
        """
        return self.peak_flops_per_s(dtype, sparsity=sparsity) / self.dram.bandwidth_bytes_per_s
