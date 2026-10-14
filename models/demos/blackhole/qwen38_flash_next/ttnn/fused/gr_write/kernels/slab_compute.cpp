// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The slab write: compute.cpp's two binary_ng programs per output tile in one 16-bit-dest kernel, the four branch
// tiles of every unit (row tile u / hidden_tiles, column tile u % hidden_tiles) per dest acquire.  ttnn.multiply
// (SFPU: the block tile copied once, coefficient column b broadcast, mul_binary_tile = fp32 product, software RNE,
// 0 * x = 0, pack bf16) then ttnn.add (FPU add_tiles(residual, update) from the two bf16 CBs, pack).  The four
// coefficient tiles of a row tile are waited at its first unit and kept for its units.
// Named compile-time args: cb_block, cb_res, cb_coef, cb_upd, cb_out, hidden_tiles, branches.  Runtime args: 0 first
// unit, 1 units.
// CONTROL (the replica's negative control): mul_binary_tile<true> stores the fp32 product unrounded (no RNE).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "../../kernels/zones.h"

using namespace ckernel;

#ifdef CONTROL
constexpr bool PRODUCT_FP32 = true;
#else
constexpr bool PRODUCT_FP32 = false;
#endif

void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t units = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_block = get_named_compile_time_arg_val("cb_block");
    constexpr uint32_t cb_res = get_named_compile_time_arg_val("cb_res");
    constexpr uint32_t cb_coef = get_named_compile_time_arg_val("cb_coef");
    constexpr uint32_t cb_upd = get_named_compile_time_arg_val("cb_upd");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t branches = get_named_compile_time_arg_val("branches");
    constexpr uint32_t dst_block = 0, dst_upd = 1;  // the block tile, then the four branch products

    compute_kernel_hw_startup(cb_block, cb_coef, cb_out);
    for (uint32_t u = first; u < first + units; ++u) {
        FUSED_ZONE("fz_gw_slab_c_unit");
        if (u == first || u % hidden_tiles == 0) {
            if (u != first) {
                cb_pop_front(cb_coef, branches);
            }
            cb_wait_front(cb_coef, branches);
        }
        cb_wait_front(cb_block, 1);
        cb_wait_front(cb_res, branches);
        cb_reserve_back(cb_upd, branches);
        mul_binary_tile_init();
        tile_regs_acquire();
        reconfig_data_format_srca(cb_block);
        copy_init(cb_block);
        copy_tile(cb_block, 0, dst_block);
        reconfig_data_format_srca(cb_coef);
        unary_bcast_init<BroadcastType::COL>(cb_coef);
        for (uint32_t b = 0; b < branches; ++b) {
            unary_bcast<BroadcastType::COL>(cb_coef, b, dst_upd + b);
        }
        for (uint32_t b = 0; b < branches; ++b) {
            mul_binary_tile<PRODUCT_FP32>(dst_block, dst_upd + b, dst_upd + b);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t b = 0; b < branches; ++b) {
            pack_tile(dst_upd + b, cb_upd);
        }
        tile_regs_release();
        cb_push_back(cb_upd, branches);
        cb_pop_front(cb_block, 1);

        cb_wait_front(cb_upd, branches);
        cb_reserve_back(cb_out, branches);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_res, cb_upd);
        tile_regs_acquire();
        for (uint32_t b = 0; b < branches; ++b) {
            add_tiles(cb_res, cb_upd, b, b, b);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t b = 0; b < branches; ++b) {
            pack_tile(b, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, branches);
        cb_pop_front(cb_upd, branches);
        cb_pop_front(cb_res, branches);
    }
    cb_pop_front(cb_coef, branches);
}
