"""Loader errors must name the file, the field, the bad value and the allowed set.

That is the M1 definition of done for validation messages (CLAUDE.md #8), so each
of those four is asserted separately rather than trusting one match.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bwz.spec import SpecLoadError, load_chip, load_deployment, load_model, profiles_root


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


VALID_CHIP = """
id: tiny
name: Tiny
vendor: Test
source_url: https://example.com/ds
clock_ghz: 1.0
compute_units:
  - {name: tc, count: 1, ops_per_cycle_per_unit: 1000, supported_dtypes: [fp16]}
memory:
  - {name: DRAM, level: 1, capacity_bytes: 1.0e+9, bandwidth_bytes_per_s: 1.0e+11}
"""


def test_loads_from_an_explicit_path(tmp_path: Path) -> None:
    path = _write(tmp_path, "tiny.yaml", VALID_CHIP)
    assert load_chip(path).name == "Tiny"


def test_error_names_file_field_value_and_bound(tmp_path: Path) -> None:
    bad = VALID_CHIP.replace("bandwidth_bytes_per_s: 1.0e+11", "bandwidth_bytes_per_s: -5.0")
    path = _write(tmp_path, "tiny.yaml", bad)
    with pytest.raises(SpecLoadError) as excinfo:
        load_chip(path)
    message = str(excinfo.value)
    assert str(path) in message, "must name the file"
    assert "memory.0.bandwidth_bytes_per_s" in message, "must name the field path"
    assert "-5.0" in message, "must show the offending value"
    assert "gt=0" in message, "must state the allowed range"


def test_missing_required_field_says_so(tmp_path: Path) -> None:
    bad = VALID_CHIP.replace("clock_ghz: 1.0\n", "")
    path = _write(tmp_path, "tiny.yaml", bad)
    with pytest.raises(SpecLoadError, match=r"clock_ghz[\s\S]*required"):
        load_chip(path)


def test_bad_enum_lists_the_allowed_values(tmp_path: Path) -> None:
    bad = VALID_CHIP.replace("supported_dtypes: [fp16]", "supported_dtypes: [float16]")
    path = _write(tmp_path, "tiny.yaml", bad)
    with pytest.raises(SpecLoadError) as excinfo:
        load_chip(path)
    message = str(excinfo.value)
    assert "float16" in message
    assert "fp16" in message and "int8" in message, "must list what is allowed"


def test_unknown_profile_id_lists_the_bundled_ones() -> None:
    with pytest.raises(SpecLoadError) as excinfo:
        load_chip("h200_sxm")
    message = str(excinfo.value)
    assert "h100_sxm" in message and "mi300x" in message


def test_unknown_model_family_names_the_allowed_set(tmp_path: Path) -> None:
    path = _write(tmp_path, "weird.yaml", "id: weird\nname: Weird\nfamily: rnn\n")
    with pytest.raises(SpecLoadError) as excinfo:
        load_model(path)
    assert "transformer_decoder" in str(excinfo.value)


def test_empty_file_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, "empty.yaml", "")
    with pytest.raises(SpecLoadError, match=r"is empty"):
        load_chip(path)


def test_non_mapping_document_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, "list.yaml", "- a\n- b\n")
    with pytest.raises(SpecLoadError, match=r"expected a mapping at the top level, got list"):
        load_chip(path)


def test_malformed_yaml_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, "bad.yaml", "name: a: b\n")
    with pytest.raises(SpecLoadError, match=r"cannot parse"):
        load_chip(path)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(SpecLoadError, match=r"no such file"):
        load_chip(tmp_path / "nope.yaml")


def test_id_must_match_filename_for_bundled_profiles(tmp_path: Path) -> None:
    """Only enforced inside the bundled directories, where id is the address."""
    path = _write(tmp_path, "renamed.yaml", VALID_CHIP)
    assert load_chip(path).id == "tiny", "outside the bundle, the id wins"


def test_deployment_config_loads(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "deploy.yaml",
        "mode: inference\nphase: decode\nbatch: 8\ninput_tokens: 512\noutput_tokens: 128\n",
    )
    spec = load_deployment(path)
    assert spec.batch == 8
    assert spec.runs_decode() and not spec.runs_prefill()


def test_deployment_error_names_the_field(tmp_path: Path) -> None:
    path = _write(tmp_path, "deploy.yaml", "batch: 0\n")
    with pytest.raises(SpecLoadError) as excinfo:
        load_deployment(path)
    assert "batch" in str(excinfo.value) and "gt=0" in str(excinfo.value)


def test_profile_path_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "chips").mkdir()
    _write(tmp_path / "chips", "tiny.yaml", VALID_CHIP)
    monkeypatch.setenv("BWZ_PROFILE_PATH", str(tmp_path))
    assert load_chip("tiny").name == "Tiny"


def test_profile_path_override_must_be_a_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BWZ_PROFILE_PATH", "/nonexistent/profiles")
    with pytest.raises(SpecLoadError, match=r"is not a directory"):
        profiles_root()
