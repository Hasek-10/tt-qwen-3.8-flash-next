# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The model's fabric router config without a device: the mesh open passes it to ``ttnn.set_fabric_config`` (on a fake
ttnn that refuses the mesh open, so the fabric is disabled again), and the demo's and the device harness's mesh
parameters carry it."""

from __future__ import annotations

from types import SimpleNamespace


def test_the_mesh_open_passes_the_model_fabric_router_config(monkeypatch, expect_error) -> None:
    """open_partition_b_mesh enables FABRIC_1D with the model's router config (8192 B packets) and disables the fabric
    when the open fails; the demo's and the device harness's mesh parameters carry the same config."""
    import ttnn
    from models.demos.blackhole.qwen38_flash_next.demo import text_demo
    from models.demos.blackhole.qwen38_flash_next.tests import tp_harness
    from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
    from models.demos.blackhole.qwen38_flash_next.ttnn import contracts

    calls = []

    def open_mesh_device(**kwargs):
        raise RuntimeError("no device in this test")

    descriptor = SimpleNamespace(local_shape=lambda: (4, 1), all_local=lambda: True)
    fake_ttnn = SimpleNamespace(
        GetNumAvailableDevices=lambda: 4,
        get_num_pcie_devices=lambda: 4,
        get_num_devices=lambda: 4,
        _ttnn=SimpleNamespace(multi_device=SimpleNamespace(SystemMeshDescriptor=lambda: descriptor)),
        FabricConfig=ttnn.FabricConfig,
        FabricReliabilityMode=ttnn.FabricReliabilityMode,
        FabricTensixConfig=ttnn.FabricTensixConfig,
        MeshShape=ttnn.MeshShape,
        set_fabric_config=lambda *args, **kwargs: calls.append((args, kwargs)),
        open_mesh_device=open_mesh_device,
    )
    profile = SimpleNamespace(
        host="host", partition="a", system_mesh_local_shape=(4, 1), route=(0, 1, 2, 3), route_nodes=(0, 1, 2, 3)
    )
    monkeypatch.setattr(session_module, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        session_module, "resolve_route", lambda hardware_profile: (hardware_profile, {"chips_with_mmio": {}})
    )
    monkeypatch.setattr(hardware_profiles, "verify_inherited_locks", lambda hardware_profile: None)

    with expect_error(RuntimeError, "no device in this test"):
        session_module.open_partition_b_mesh(lambda phase: None, profile)

    (enable_args, enable_kwargs), (disable_args, disable_kwargs) = calls
    assert enable_args == (
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    assert (
        enable_kwargs["router_config"].max_packet_payload_size_bytes
        == contracts.FABRIC_MAX_PACKET_PAYLOAD_BYTES
        == 8192
    )
    assert disable_args == (ttnn.FabricConfig.DISABLED,) and disable_kwargs == {}
    for params in (text_demo.DEVICE_PARAMS, tp_harness.DEVICE_PARAMS):
        assert params["fabric_router_config"].max_packet_payload_size_bytes == 8192
