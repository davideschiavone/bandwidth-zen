"""Capacity planning: what must fit, whether it does, and what SRAM buys.

``docs/MODEL.md`` §6.2. Two jobs, both flowing from capacity and neither from
bandwidth (``docs/CORRECTIONS.md`` D5a):

1. **Feasibility.** Weights + KV cache + peak activations against usable DRAM. An
   infeasible configuration returns a ``Report`` with ``feasible: false`` and the
   cheapest fixes — never an exception, never a 500 (CLAUDE.md #8).
2. **Residency.** ``r = min(1, on_chip / W)`` is the weight fraction that need not
   be re-streamed from DRAM each token, and it is the only way SRAM enters the
   timing model.
"""

from __future__ import annotations

from bwz.analysis.tiling import double_buffering_fits, tile_bytes
from bwz.calibration import DEFAULT_USABLE_MEMORY_FRACTION
from bwz.graph.ops import ComputeGraph
from bwz.report import MemoryPlan
from bwz.spec.deployment import DeploymentSpec
from bwz.spec.dtypes import bytes_per_element
from bwz.spec.hardware_spec import HardwareSpec
from bwz.units import format_bytes


def usable_memory_fraction(chip: HardwareSpec) -> float:
    """Per-chip override if the profile carries one, else the calibrated default (D6)."""
    if chip.usable_memory_fraction is not None:
        return chip.usable_memory_fraction
    return DEFAULT_USABLE_MEMORY_FRACTION


def plan_memory(graph: ComputeGraph, chip: HardwareSpec, deployment: DeploymentSpec) -> MemoryPlan:
    """Size the footprint and compute residency and double-buffering headroom.

    ``peak_activation_bytes`` comes from the graph's liveness analysis, which
    assumes an allocator that frees a tensor the instant its last reader
    completes. That is optimistic, and it is recorded as an assumption.
    """
    from bwz.graph.dag import peak_activation_bytes

    weight_bytes = graph.weight_footprint_bytes()
    kv_bytes = graph.kv_cache_bytes()
    activation_bytes = peak_activation_bytes(graph)
    total = weight_bytes + kv_bytes + activation_bytes

    fraction = usable_memory_fraction(chip)
    dram_capacity = chip.dram.capacity_bytes
    usable = dram_capacity * fraction

    on_chip = chip.on_chip_capacity_bytes
    unit = max(chip.compute_units, key=lambda u: u.count * u.ops_per_cycle_per_unit)
    element_bytes = bytes_per_element(deployment.precision.weights)
    working_tile = tile_bytes(unit, element_bytes)

    # On-chip capacity is allocated in the order that saves the most DRAM
    # traffic, which is what an NPU compiler does and what the D8 formulas
    # implicitly assume:
    #
    #   1. The double buffer. Filling SRAM to the brim with resident weights
    #      leaves no staging room, forcing loads to serialise behind compute --
    #      a far worse trade than giving up two tiles of residency. chip_b at
    #      1 GB is the case that surfaces it.
    #   2. The activation working set. Activations are read and written many
    #      times within a phase and are small; keeping them on chip is almost
    #      always the better use of a byte than weight residency. On chip_a at
    #      prefill, 15 MB of activations would otherwise cost 2 GB of DRAM
    #      traffic -- 58 ms against the 1.6 ms that spending the same capacity
    #      on weight residency would save.
    #   3. Weights, with whatever is left. This is what the residency fraction
    #      measures.
    available = max(0.0, on_chip - 2.0 * working_tile)
    activation_resident_fraction = (
        min(1.0, available / activation_bytes) if activation_bytes > 0 else 1.0
    )
    available -= activation_resident_fraction * activation_bytes

    resident_fraction = min(1.0, available / weight_bytes) if weight_bytes > 0 else 1.0
    resident_bytes = resident_fraction * weight_bytes
    buffered = double_buffering_fits(
        on_chip, resident_bytes + activation_resident_fraction * activation_bytes, working_tile
    )

    return MemoryPlan(
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_bytes,
        peak_activation_bytes=activation_bytes,
        total_bytes=total,
        dram_capacity_bytes=dram_capacity,
        usable_dram_bytes=usable,
        on_chip_capacity_bytes=on_chip,
        resident_fraction=resident_fraction,
        activation_resident_fraction=activation_resident_fraction,
        double_buffered=buffered,
        fits=total <= usable,
    )


def infeasibility_reasons(
    plan: MemoryPlan, chip: HardwareSpec, deployment: DeploymentSpec
) -> tuple[str, ...]:
    """Why the configuration cannot run, and the cheapest fixes.

    "Cheapest" is ordered by how little the user gives up: precision first (a
    weight-dtype change is free at inference time and halves the dominant term),
    then context, then batch, then hardware. Each fix names the number it would
    have to reach, so it can be acted on without a second run.
    """
    if plan.fits:
        return ()

    over_by = plan.total_bytes - plan.usable_dram_bytes
    reasons = [
        f"Does not fit: needs {format_bytes(plan.total_bytes)} "
        f"({format_bytes(plan.weight_bytes)} weights + {format_bytes(plan.kv_cache_bytes)} KV "
        f"cache + {format_bytes(plan.peak_activation_bytes)} activations) against "
        f"{format_bytes(plan.usable_dram_bytes)} usable of "
        f"{format_bytes(plan.dram_capacity_bytes)} on {chip.name}. "
        f"Over by {format_bytes(over_by)}."
    ]

    weight_element_bytes = bytes_per_element(deployment.precision.weights)
    if weight_element_bytes > 1.0:
        saved = plan.weight_bytes / 2.0
        verdict = "would fit" if saved >= over_by else "is not enough on its own"
        reasons.append(
            f"Cheapest fix: halve the weight precision (currently "
            f"{deployment.precision.weights.value}), saving {format_bytes(saved)} — {verdict}."
        )

    if plan.kv_cache_bytes > over_by and deployment.context_tokens > 1:
        share = 1.0 - over_by / plan.kv_cache_bytes
        reasons.append(
            f"Or cut the context from {deployment.context_tokens} tokens to about "
            f"{int(deployment.context_tokens * share)}, saving {format_bytes(over_by)} of KV cache."
        )

    if deployment.batch > 1:
        reasons.append(
            f"Or reduce batch from {deployment.batch}; activations and KV cache both scale with it."
        )

    chips_needed = max(2, int(-(-plan.total_bytes // plan.usable_dram_bytes)))
    reasons.append(
        f"Or shard across {chips_needed} chips (tensor parallelism lands at M5; v1 is single-chip)."
    )
    return tuple(reasons)
