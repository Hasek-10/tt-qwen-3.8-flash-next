# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused decode kernels over ``ttnn.generic_op``.

``program`` builds and runs one program from Python (kernels, CBs, semaphores, the rows contract); ``registry`` names
each fused kernel with the composed ttnn chain it replaces and switches it on through ``QWEN38_FUSED``.  Each kernel
is a sub-package ``<name>/`` with its ``kernels/*.cpp`` and registers itself on import; add new ones to the import
list below.  Gate and accounting: FUSED-KERNEL-HOWTO.md under the dev tools.
"""

from . import program
from .registry import (
    ALL,
    BITWISE,
    COMPONENT,
    ENV,
    TOLERANCE_CLASSES,
    ULP,
    FusedKernel,
    GateSpec,
    enabled,
    enabled_names,
    kernel,
    kernels,
    register,
    resolve,
)
from . import router_tail, untilize_rows

__all__ = [
    "ALL",
    "BITWISE",
    "COMPONENT",
    "ENV",
    "TOLERANCE_CLASSES",
    "ULP",
    "FusedKernel",
    "GateSpec",
    "enabled",
    "enabled_names",
    "kernel",
    "kernels",
    "program",
    "register",
    "resolve",
    "router_tail",
    "untilize_rows",
]
