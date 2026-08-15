"""YAML/JSON to validated spec, with error messages a user can act on.

This is the only module permitted to traffic in ``Any`` (CLAUDE.md style): it is
the boundary between untyped user files and the typed core.

A validation failure names the file, the field path, the offending value and the
allowed range or set (CLAUDE.md #8). Compare::

    bwz: cannot load chip profile 'h100_sxm' (backend/profiles/chips/h100_sxm.yaml)
      memory.2.bandwidth_bytes_per_s: Input should be greater than 0
        got: -3350000000000.0

against pydantic's default multi-line dump. Errors are raised as
:class:`SpecLoadError`, which the CLI prints without a traceback.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

from bwz.spec.deployment import DeploymentSpec
from bwz.spec.hardware_spec import HardwareSpec
from bwz.spec.model_spec import CNNSpec, CustomSpec, GemmSpec, ModelSpec, TransformerSpec

_PROFILE_ENV_VAR = "BWZ_PROFILE_PATH"

AnyModelSpec = TransformerSpec | CNNSpec | GemmSpec | CustomSpec
_MODEL_ADAPTER: TypeAdapter[AnyModelSpec] = TypeAdapter(ModelSpec)


class SpecLoadError(Exception):
    """A profile or config could not be read or validated. Message is user-facing."""


# -- profile discovery ------------------------------------------------------


def profiles_root() -> Path:
    """Directory holding ``chips/`` and ``models/``.

    Resolution order: ``$BWZ_PROFILE_PATH``, then the packaged copy at
    ``bwz/profiles`` (populated by hatch ``force-include`` on install), then the
    repo checkout at ``backend/profiles``. The last case is what a
    ``uv run`` development session hits.
    """
    override = os.environ.get(_PROFILE_ENV_VAR)
    if override:
        path = Path(override).expanduser()
        if not path.is_dir():
            raise SpecLoadError(
                f"{_PROFILE_ENV_VAR}={override!r} is not a directory; unset it or point it at a "
                f"directory containing chips/ and models/"
            )
        return path
    here = Path(__file__).resolve()
    packaged = here.parent.parent / "profiles"
    if packaged.is_dir():
        return packaged
    repo = here.parents[2] / "profiles"
    if repo.is_dir():
        return repo
    raise SpecLoadError(
        f"cannot locate bundled profiles; looked in {packaged} and {repo}. Set "
        f"{_PROFILE_ENV_VAR} to a directory containing chips/ and models/."
    )


def _profile_dir(kind: str) -> Path:
    return profiles_root() / kind


def available_chips() -> list[str]:
    """Ids of the bundled chip profiles, sorted."""
    return _available(_profile_dir("chips"))


def available_models() -> list[str]:
    """Ids of the bundled model profiles, sorted."""
    return _available(_profile_dir("models"))


def _available(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


# -- raw file reading -------------------------------------------------------


def _read_document(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecLoadError(f"cannot read {path}: {exc}") from exc
    try:
        loaded: Any = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise SpecLoadError(f"cannot parse {path}: {exc}") from exc
    if loaded is None:
        raise SpecLoadError(f"{path} is empty")
    if not isinstance(loaded, dict):
        raise SpecLoadError(
            f"{path}: expected a mapping at the top level, got {type(loaded).__name__}"
        )
    return loaded


def _resolve(kind: str, ref: str | Path) -> Path:
    """Turn a profile id or filesystem path into a readable path."""
    as_path = Path(ref)
    if as_path.suffix in (".yaml", ".yml", ".json") or as_path.exists():
        if not as_path.is_file():
            raise SpecLoadError(f"no such file: {as_path}")
        return as_path
    directory = _profile_dir(kind)
    candidate = directory / f"{ref}.yaml"
    if candidate.is_file():
        return candidate
    known = _available(directory)
    raise SpecLoadError(
        f"unknown {kind[:-1]} profile {str(ref)!r}; bundled profiles are {known}. "
        f"Pass a path to a YAML file to use your own."
    )


# -- error formatting -------------------------------------------------------


def _location(error: dict[str, Any]) -> str:
    parts: Iterable[Any] = error.get("loc", ())
    rendered = ".".join(str(p) for p in parts)
    return rendered or "<root>"


def _allowed(error: dict[str, Any]) -> str:
    """Render the permitted values or bounds for an error, when pydantic supplies them."""
    ctx: dict[str, Any] = error.get("ctx", {}) or {}
    if "expected" in ctx:
        return f" allowed: {ctx['expected']}"
    bounds = [
        (key, ctx[key])
        for key in ("gt", "ge", "lt", "le", "min_length", "max_length")
        if key in ctx
    ]
    if bounds:
        return " allowed: " + ", ".join(f"{k}={v}" for k, v in bounds)
    return ""


def _format_errors(exc: ValidationError, path: Path, what: str) -> str:
    lines = [f"cannot load {what} ({path})"]
    for raw in exc.errors():
        # exc.errors() yields a TypedDict; a plain dict is what the helpers below want.
        error: dict[str, Any] = dict(raw)
        location = _location(error)
        message = error.get("msg", "invalid value")
        lines.append(f"  {location}: {message}")
        if error.get("type") not in ("value_error", "missing"):
            lines.append(f"    got: {error.get('input')!r}{_allowed(error)}")
        elif error.get("type") == "missing":
            lines.append("    this field is required")
    return "\n".join(lines)


def _validate(adapter: TypeAdapter[Any], document: dict[str, Any], path: Path, what: str) -> Any:
    try:
        return adapter.validate_python(document)
    except ValidationError as exc:
        raise SpecLoadError(_format_errors(exc, path, what)) from exc


# -- public loaders ---------------------------------------------------------


def load_chip(ref: str | Path) -> HardwareSpec:
    """Load a chip profile by bundled id (``"h100_sxm"``) or path."""
    path = _resolve("chips", ref)
    document = _read_document(path)
    document.setdefault("id", path.stem)
    spec: HardwareSpec = _validate(
        TypeAdapter(HardwareSpec), document, path, f"chip profile {path.stem!r}"
    )
    _check_id_matches_filename(spec.id, path, "chip")
    return spec


def load_model(ref: str | Path) -> AnyModelSpec:
    """Load a model profile by bundled id (``"llama3_8b"``) or path."""
    path = _resolve("models", ref)
    document = _read_document(path)
    document.setdefault("id", path.stem)
    spec: AnyModelSpec = _validate(_MODEL_ADAPTER, document, path, f"model profile {path.stem!r}")
    _check_id_matches_filename(spec.id, path, "model")
    return spec


def load_deployment(ref: str | Path) -> DeploymentSpec:
    """Load a deployment config from a YAML/JSON file."""
    path = Path(ref)
    if not path.is_file():
        raise SpecLoadError(f"no such deployment config: {path}")
    document = _read_document(path)
    spec: DeploymentSpec = _validate(
        TypeAdapter(DeploymentSpec), document, path, "deployment config"
    )
    return spec


def _check_id_matches_filename(spec_id: str, path: Path, kind: str) -> None:
    if path.parent == _profile_dir(f"{kind}s") and spec_id != path.stem:
        raise SpecLoadError(
            f"{kind} profile {path}: id {spec_id!r} does not match the filename stem "
            f"{path.stem!r}; bundled profiles are addressed by filename, so the two must agree"
        )


def iter_chips() -> Iterator[HardwareSpec]:
    """Every bundled chip profile, in id order."""
    for chip_id in available_chips():
        yield load_chip(chip_id)


def iter_models() -> Iterator[AnyModelSpec]:
    """Every bundled model profile, in id order."""
    for model_id in available_models():
        yield load_model(model_id)


# -- round-tripping ---------------------------------------------------------


def to_document(spec: BaseModel) -> dict[str, Any]:
    """Serialise a spec back to a plain dict suitable for YAML.

    ``exclude_defaults`` is deliberately off: the dumped document is the *fully
    resolved* spec, which is what round-trip tests compare. Textual identity with
    the source YAML is not a goal — comments, key order and human units
    (``"3.35 TB/s"`` -> ``3.35e12``) are lost by design.
    """
    dumped: dict[str, Any] = spec.model_dump(mode="json", exclude_none=True)
    return dumped


def to_yaml(spec: BaseModel) -> str:
    """Dump a spec to YAML text."""
    return yaml.safe_dump(to_document(spec), sort_keys=False, default_flow_style=False)
