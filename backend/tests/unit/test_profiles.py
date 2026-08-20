"""Every shipped profile loads, round-trips and carries its provenance.

Round-trip is **spec -> document -> spec**, not textual: comments, key order and
human units (``"3.35 TB/s"`` -> ``3.35e12``) are lost by design (see
``loaders.to_document``).
"""

from __future__ import annotations

import pytest
import yaml

from bwz.analysis import machine_model
from bwz.spec import (
    DType,
    HardwareSpec,
    available_chips,
    available_models,
    load_chip,
    load_model,
    profiles_root,
    to_document,
    to_yaml,
)
from bwz.spec.loaders import _MODEL_ADAPTER, AnyModelSpec


def test_the_expected_profiles_ship() -> None:
    """M1 definition of done: six chips, eight models. Metis joins as the seventh
    chip and the first profile whose numbers come from a paper rather than a
    datasheet."""
    assert available_chips() == [
        "a100_80gb",
        "chip_a",
        "chip_b",
        "h100_sxm",
        "jetson_orin",
        "metis_aipu",
        "mi300x",
    ]
    assert available_models() == [
        "gemma3_4b",
        "gemma3_preset_1b",
        "gemma3_preset_2b",
        "gpt3",
        "llama2_70b",
        "llama3_8b",
        "mistral_7b",
        "mobilenetv3",
        "single_layer_encoder_toy",
    ]


@pytest.mark.parametrize("chip_id", available_chips())
def test_chip_round_trips(chip_id: str) -> None:
    original = load_chip(chip_id)
    reloaded = HardwareSpec.model_validate(yaml.safe_load(to_yaml(original)))
    assert reloaded == original


@pytest.mark.parametrize("model_id", available_models())
def test_model_round_trips(model_id: str) -> None:
    original = load_model(model_id)
    reloaded: AnyModelSpec = _MODEL_ADAPTER.validate_python(yaml.safe_load(to_yaml(original)))
    assert reloaded == original


@pytest.mark.parametrize("chip_id", available_chips())
def test_chip_provenance(chip_id: str) -> None:
    """CLAUDE.md: never commit profile YAML without a source_url. A profile that
    cannot cite one must declare itself hypothetical (docs/CORRECTIONS.md D7)."""
    chip = load_chip(chip_id)
    assert chip.source_url is not None or chip.hypothetical
    if chip.hypothetical:
        assert chip.estimates, f"{chip_id}: a hypothetical profile must explain itself"


@pytest.mark.parametrize("model_id", available_models())
def test_model_provenance(model_id: str) -> None:
    model = load_model(model_id)
    assert model.source_url is not None or model.hypothetical


@pytest.mark.parametrize("chip_id", available_chips())
def test_chip_id_matches_filename(chip_id: str) -> None:
    assert load_chip(chip_id).id == chip_id


@pytest.mark.parametrize("model_id", available_models())
def test_model_id_matches_filename(model_id: str) -> None:
    assert load_model(model_id).id == model_id


@pytest.mark.parametrize("chip_id", available_chips())
def test_chip_memory_is_ordered_and_dram_is_slowest(chip_id: str) -> None:
    """The deepest level must be the slowest, or ``dram`` is picking the wrong one."""
    chip = load_chip(chip_id)
    bandwidths = [m.bandwidth_bytes_per_s for m in chip.memory]
    assert bandwidths == sorted(bandwidths, reverse=True), chip_id
    assert chip.dram.bandwidth_bytes_per_s == min(bandwidths)


@pytest.mark.parametrize("chip_id", available_chips())
def test_chip_capacity_grows_with_depth(chip_id: str) -> None:
    chip = load_chip(chip_id)
    capacities = [m.capacity_bytes for m in chip.memory]
    assert capacities == sorted(capacities), chip_id


def test_no_stray_files_in_the_profile_directories() -> None:
    """A ``.yml`` or ``.json`` profile would be invisible to ``available_*``."""
    for kind in ("chips", "models"):
        directory = profiles_root() / kind
        stray = sorted(p.name for p in directory.iterdir() if p.suffix != ".yaml")
        assert stray == [], f"{kind}: only .yaml profiles are discovered, found {stray}"


def test_metis_aipu_reproduces_the_published_figures() -> None:
    """Golden for the ISSCC 2024 profile, hand-computed from the paper.

    P. A. Hager et al., "11.3 Metis AIPU: A 12nm 15TOPS/W 209.6TOPS SoC for Cost-
    and Energy-Efficient Inference at the Edge", ISSCC 2024, pp. 212-213.

    **Peak.** Fig 11.3.4(a) gives a 512x512 INT8 D-IMC array per AI core, bit
    serial with "accumulation over 8 input bit cycles"; Table C gives four AI
    cores at 800 MHz::

        512 x 512                     = 262144 MAC/cycle/core
        / 8                           =  32768 MAC/cycle/core  (bit-serial tax)
        x 0.8e9 x 2 OP/MAC            =  5.242880e13 OP/s      = 52.4288 TOPS/core
        x 4 cores                     =  2.097152e14 OP/s      = 209.7152 TOPS

    against the paper's own 52.4 TOPS/core (Table B and Table C) and 209.6 TOPS
    in the title, which is 4 x the rounded per-core figure. The two agree to
    0.06%, and the 0.06% is that rounding.

    **Ridge point.** 2.097152e14 / 3.4128e10 = 6144.96 OP/byte. Enormous, and
    that is the honest reading of an edge NPU with 210 TOPS behind a 64-bit
    LPDDR4x bus: essentially every real layer lands left of the ridge. Note the
    denominator is an estimate, not a published figure (see the profile's
    ``estimates.memory``) — this golden pins the arithmetic, not the datasheet.

    **On-chip capacity.** 4 x 1 MiB D-IMC + 4 x 4 MiB L1 + 32 MiB L2 = 52 MiB,
    which is the total the paper states in prose: "Including the 4 MiB L1 and the
    1 MiB D-IMC SRAM of each of the four AI-Cores, the on-chip SRAM aggregates to
    52 MiB."
    """
    chip = load_chip("metis_aipu")

    assert chip.peak_flops_per_s(DType.INT8) == pytest.approx(2.097152e14, rel=1e-12)
    assert chip.peak_flops_per_s(DType.INT8) / 1e12 == pytest.approx(209.6, rel=6e-4)
    assert chip.ridge_point_flops_per_byte(DType.INT8) == pytest.approx(6144.96, rel=1e-4)
    assert chip.on_chip_capacity_bytes == 52 * 1024**2

    # The paper's measured peak, as a second point on the same line: 875 MHz at
    # 0.7 V gives 57.344 TOPS/core against the stated 57.3. `peak_flops_per_s`
    # folds in `count`, so divide by the four cores to get the per-core figure
    # the paper quotes.
    array = next(u for u in chip.compute_units if u.name == "d_imc")
    per_core = array.peak_flops_per_s(0.875e9, DType.INT8) / array.count
    assert per_core / 1e12 == pytest.approx(57.3, rel=1e-3)
    assert array.peak_flops_per_s(chip.clock_hz, DType.INT8) / array.count / 1e12 == pytest.approx(
        52.4, rel=1e-3
    )

    # A vector unit distinct from the matrix engine (docs/CORRECTIONS.md D27):
    # the DPU exists on this chip, so norms and activations are not charged at
    # the array's rate the way they are on chip_a.
    machine = machine_model(chip, DType.INT8)
    assert machine.has_vector_unit
    assert (machine.unit.name, machine.vector_unit.name) == ("d_imc", "dpu")

    # Four declared levels against the flat v1 roofline's two (D5): the figure
    # draws the middle ones grey rather than omitting them (D20).
    assert [m.name for m in chip.memory] == ["D-IMC", "L1", "L2", "LPDDR4x"]

    # It describes real, measured silicon, so it cites rather than declaring
    # itself hypothetical — and every inferred field names itself (D6/D7).
    assert chip.source_url is not None and not chip.hypothetical
    assert set(chip.estimates) == {
        "memory",
        "compute_units",
        "dram_bandwidth_efficiency",
        "achieved_flops_fraction",
        "tdp_w",
        "kernel_launch_overhead_s",
    }


def test_to_document_omits_nothing_that_matters() -> None:
    """A dumped chip must carry enough to rebuild it; None fields are dropped."""
    document = to_document(load_chip("chip_a"))
    assert document["id"] == "chip_a"
    assert "source_url" not in document, "chip_a has none; exclude_none should drop it"
    assert document["hypothetical"] is True
