"""Every shipped profile loads, round-trips and carries its provenance.

Round-trip is **spec -> document -> spec**, not textual: comments, key order and
human units (``"3.35 TB/s"`` -> ``3.35e12``) are lost by design (see
``loaders.to_document``).
"""

from __future__ import annotations

import pytest
import yaml

from bwz.spec import (
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
    """M1 definition of done: six chips, eight models."""
    assert available_chips() == [
        "a100_80gb",
        "chip_a",
        "chip_b",
        "h100_sxm",
        "jetson_orin",
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
        "single_layer_encoder",
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


def test_to_document_omits_nothing_that_matters() -> None:
    """A dumped chip must carry enough to rebuild it; None fields are dropped."""
    document = to_document(load_chip("chip_a"))
    assert document["id"] == "chip_a"
    assert "source_url" not in document, "chip_a has none; exclude_none should drop it"
    assert document["hypothetical"] is True
