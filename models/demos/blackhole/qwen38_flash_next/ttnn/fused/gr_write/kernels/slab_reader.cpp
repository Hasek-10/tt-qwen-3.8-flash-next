// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The slab write's reader.  Unit u (row tile u / hidden_tiles, column tile u % hidden_tiles) of the core's consecutive
// units: the block tile u and the residual tiles branch * slab_units + u of the four branches.  At the first unit of
// each row tile r the four coefficient tiles of r: injection tile r once per branch b with its column b copied into
// column 0 (rows 0..31, reader.cpp's patch), which the compute column-broadcasts for every unit of r.
// Named compile-time args: cb_block, cb_res, cb_coef, hidden_tiles, branches.  Compile-time args: TensorAccessorArgs
// for block, residual, injection, chained from 0.
// Runtime args: 0 block addr, 1 residual addr, 2 injection addr, 3 first unit, 4 units, 5 slab_units (row tiles x
// hidden_tiles: the residual's pages per branch).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t block_addr = get_arg_val<uint32_t>(0);
    const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t injection_addr = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t units = get_arg_val<uint32_t>(4);
    const uint32_t slab_units = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_block = get_named_compile_time_arg_val("cb_block");
    constexpr uint32_t cb_res = get_named_compile_time_arg_val("cb_res");
    constexpr uint32_t cb_coef = get_named_compile_time_arg_val("cb_coef");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t branches = get_named_compile_time_arg_val("branches");
    constexpr auto block_args = TensorAccessorArgs<0>();
    constexpr auto residual_args = TensorAccessorArgs<block_args.next_compile_time_args_offset()>();
    constexpr auto injection_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();

    const auto block = TensorAccessor(block_args, block_addr);
    const auto residual = TensorAccessor(residual_args, residual_addr);
    const auto injection = TensorAccessor(injection_args, injection_addr);
    const uint32_t tile_bytes = get_tile_size(cb_block);
    Noc noc;
    DataflowBuffer blk(cb_block);
    DataflowBuffer res(cb_res);
    DataflowBuffer coef(cb_coef);

    for (uint32_t u = first; u < first + units; ++u) {
        FUSED_ZONE("fz_gw_slab_r_unit");
        if (u == first || u % hidden_tiles == 0) {
            coef.reserve_back(branches);
            for (uint32_t b = 0; b < branches; ++b) {
                noc.async_read(
                    injection, coef, tile_bytes, {.page_id = u / hidden_tiles}, {.offset_bytes = b * tile_bytes});
            }
            noc.async_read_barrier();
            // bf16 tile element (r, c) is 16-bit word (r >> 4) * 512 + (c >> 4) * 256 + (r & 15) * 16 + (c & 15)
            for (uint32_t b = 1; b < branches; ++b) {
                volatile tt_l1_ptr uint16_t* tile =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(coef.get_write_ptr() + b * tile_bytes);
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t row0 = (r >> 4) * 512 + (r & 15) * 16;
                    tile[row0] = tile[row0 + b];
                }
            }
            coef.push_back(branches);
        }
        blk.reserve_back(1);
        res.reserve_back(branches);
        noc.async_read(block, blk, tile_bytes, {.page_id = u}, {.offset_bytes = 0});
        for (uint32_t b = 0; b < branches; ++b) {
            noc.async_read(
                residual, res, tile_bytes, {.page_id = b * slab_units + u}, {.offset_bytes = b * tile_bytes});
        }
        noc.async_read_barrier();
        blk.push_back(1);
        res.push_back(branches);
    }
}
