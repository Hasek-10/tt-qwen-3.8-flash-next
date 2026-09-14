# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused decode kernels by name and the ``QWEN38_FUSED`` / ``QWEN38_FUSED_OFF`` switches.

Every fused kernel stands in for one named chain of existing ttnn ops.  A kernel registered ``default_on`` (proven
bitwise against its chain and at the model level) serves by default; ``QWEN38_FUSED_OFF`` (comma-separated names, or
``all``) falls back to the composed chains, ``QWEN38_FUSED`` switches an opt-in kernel on.  A name that is not
registered raises in either variable, so a typo cannot silently change what runs.  Callers resolve once at
construction and keep the choice through trace capture: ``run = resolve("router_tail")``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

ENV = "QWEN38_FUSED"
OFF_ENV = "QWEN38_FUSED_OFF"
ALL = "all"
# Tolerance classes of the component gate (fused_component_gate, dev tools): BITWISE where the arithmetic order is preserved (pure data
# movement, integer work, the same ops in the same order); ULP where the same math is re-associated or fused
# (an eltwise chain in one pass, a reduction in another order); COMPONENT where the chain's precision points move
# (fp32 state kept, a rounding removed) and only the tt/ oracle can judge the component.
BITWISE, ULP, COMPONENT = "bitwise", "ulp", "component"
TOLERANCE_CLASSES = (BITWISE, ULP, COMPONENT)
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class GateSpec:
    """How the component gate feeds a kernel captured real inputs (``numerics_audit_device.py`` records).

    ``inputs(mesh, capture, positions, layer)`` returns the keyword arguments both callables take, as device tensors
    built from the capture's positions (one tile row per position); ``output(result)`` returns the call's result as a
    host tensor ``[rows, ...]``; ``reference(oracle, positions, layer)`` returns the oracle's value of the same
    quantity (or None when the oracle keeps none).  ``topk`` names the ranking width when the output is a ranking.
    """

    inputs: Callable[[Any, Mapping[str, Any], tuple[int, ...], int], dict[str, Any]]
    output: Callable[[Any], Any]
    reference: Callable[[Mapping[str, Any], tuple[int, ...], int], Any] | None = None
    layers: tuple[int, ...] = (0,)
    topk: int | None = None


@dataclass(frozen=True)
class FusedKernel:
    name: str
    replaces: str
    tolerance: str
    fused: Callable[..., Any]
    composed: Callable[..., Any]
    gate: GateSpec | None = None
    default_on: bool = False  # serves by default (QWEN38_FUSED_OFF falls back); False = opt-in through QWEN38_FUSED

    def __post_init__(self) -> None:
        if not _NAME.match(self.name) or self.name == ALL:
            raise ValueError(f"fused kernel name must match {_NAME.pattern} and not be {ALL!r}, got {self.name!r}")
        if self.tolerance not in TOLERANCE_CLASSES:
            raise ValueError(
                f"fused kernel {self.name}: tolerance must be one of {TOLERANCE_CLASSES}, got {self.tolerance!r}"
            )


_REGISTRY: dict[str, FusedKernel] = {}


def register(kernel: FusedKernel) -> FusedKernel:
    if kernel.name in _REGISTRY:
        raise ValueError(f"fused kernel {kernel.name!r} is registered twice")
    _REGISTRY[kernel.name] = kernel
    return kernel


def kernels() -> dict[str, FusedKernel]:
    return dict(_REGISTRY)


def kernel(name: str) -> FusedKernel:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no fused kernel {name!r}; registered: {sorted(_REGISTRY)}") from None


def _names(environ: Mapping[str, str], variable: str) -> frozenset[str]:
    tokens = [t.strip() for t in environ.get(variable, "").split(",") if t.strip()]
    if ALL in tokens:
        return frozenset(_REGISTRY)
    unknown = sorted(set(tokens) - set(_REGISTRY))
    if unknown:
        raise ValueError(f"{variable} names unregistered fused kernels {unknown}; registered: {sorted(_REGISTRY)}")
    return frozenset(tokens)


def default_names() -> frozenset[str]:
    return frozenset(name for name, entry in _REGISTRY.items() if entry.default_on)


def enabled_names(environ: Mapping[str, str] = os.environ) -> frozenset[str]:
    """The kernels that run: the ``default_on`` ones plus ``QWEN38_FUSED``, less ``QWEN38_FUSED_OFF``; every name in
    either variable must be registered."""

    return (default_names() | _names(environ, ENV)) - _names(environ, OFF_ENV)


def enabled(name: str, environ: Mapping[str, str] = os.environ) -> bool:
    kernel(name)
    return name in enabled_names(environ)


def resolve(name: str, environ: Mapping[str, str] = os.environ) -> Callable[..., Any]:
    """The kernel's fused callable when it runs (see ``enabled_names``), else its composed chain."""

    entry = kernel(name)
    return entry.fused if name in enabled_names(environ) else entry.composed
