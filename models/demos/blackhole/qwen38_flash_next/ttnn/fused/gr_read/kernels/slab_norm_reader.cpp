// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The slab normalize program's reader: reader.cpp's streams in the compute's order (the core's gathered stats tiles,
// its first row block, the branch's gamma tiles, its other row blocks) and constants (the AVG scaler, the eps column
// scalar), with the gamma tiles read from DRAM once per NoC column segment of a branch's cores: the segment's
// sender (its top core, the start corner of a NOC_0 multicast down the column) reads them into its gamma CB and
// multicasts the same bytes to the CB (one address on every core) of the cores below it, then raises their semaphore;
// a receiver waits for the semaphore before it pushes the CB.
// Compile-time args: 0 stats cb, 1 residual cb, 2 gamma cb, 3 scaler cb, 4 eps cb, 5 semaphore id,
//   6.. three TensorAccessorArgs sets (stats, residual, gamma).
// Runtime args: 0-2 stats (addr, first, count), 3-5 residual (addr, first, count), 6-7 gamma (addr, first),
//   8 sender (1) or receiver (0), 9 the sender's receivers, 10-12 their NoC x, first y, last y, 13 scaler bits,
//   14 eps bits.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "../../kernels/zones.h"

constexpr uint32_t C_STATS = get_compile_time_arg_val(0);
constexpr uint32_t C_RES = get_compile_time_arg_val(1);
constexpr uint32_t C_GAMMA = get_compile_time_arg_val(2);
constexpr uint32_t C_SCALER = get_compile_time_arg_val(3);
constexpr uint32_t C_EPS = get_compile_time_arg_val(4);
constexpr uint32_t SEM_ID = get_compile_time_arg_val(5);
constexpr uint32_t ACCESSOR_BASE = 6;
constexpr uint32_t BATCH = 4;

template <typename Args>
FORCE_INLINE void read_run(
    const Noc& noc, const Args& args, uint32_t addr, uint32_t cb, uint32_t first, uint32_t count) {
    const auto accessor = TensorAccessor(args, addr);
    const uint32_t tile_bytes = get_tile_size(cb);
    DataflowBuffer dfb(cb);
    for (uint32_t t = 0; t < count; t += BATCH) {
        dfb.reserve_back(BATCH);
        for (uint32_t i = 0; i < BATCH; ++i) {
            noc.async_read(accessor, dfb, tile_bytes, {.page_id = first + t + i}, {.offset_bytes = i * tile_bytes});
        }
        noc.async_read_barrier();
        dfb.push_back(BATCH);
    }
}

void kernel_main() {
    constexpr auto stats_args = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto res_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<res_args.next_compile_time_args_offset()>();
    const uint32_t res_addr = get_arg_val<uint32_t>(3);
    const uint32_t res_first = get_arg_val<uint32_t>(4);
    const uint32_t res_count = get_arg_val<uint32_t>(5);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    const uint32_t gamma_first = get_arg_val<uint32_t>(7);
    const bool sender = get_arg_val<uint32_t>(8) == 1;
    const uint32_t receivers = get_arg_val<uint32_t>(9);
    constexpr uint32_t Wt = 20;
    Noc noc;
    {
        FUSED_ZONE("fz_gr_slab_rd_consts");
        dataflow_kernel_lib::prepare_reduce_scaler<C_SCALER, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            __builtin_bit_cast(float, get_arg_val<uint32_t>(13)));
        generate_bcast_col_scalar(CircularBuffer(C_EPS), get_arg_val<uint32_t>(14));
    }
    {
        FUSED_ZONE("fz_gr_slab_rd_first");
        read_run(
            noc, stats_args, get_arg_val<uint32_t>(0), C_STATS, get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2));
        read_run(noc, res_args, res_addr, C_RES, res_first, Wt);
    }
    {
        FUSED_ZONE("fz_gr_slab_rd_gamma");
        DataflowBuffer gamma(C_GAMMA);
        Semaphore<> ready(SEM_ID);
        gamma.reserve_back(Wt);
        if (!sender) {  // the sender's multicast lands in this CB
            ready.wait(1);
            ready.set(0);
        } else {
            const auto accessor = TensorAccessor(gamma_args, gamma_addr);
            const uint32_t tile_bytes = get_tile_size(C_GAMMA);
            for (uint32_t j = 0; j < Wt; ++j) {
                noc.async_read(
                    accessor, gamma, tile_bytes, {.page_id = gamma_first + j}, {.offset_bytes = j * tile_bytes});
            }
            noc.async_read_barrier();
            if (receivers > 0) {
                const uint32_t x = get_arg_val<uint32_t>(10);
                const uint32_t y0 = get_arg_val<uint32_t>(11);
                const uint32_t y1 = get_arg_val<uint32_t>(12);
                const uint32_t addr = gamma.get_write_ptr();
                MulticastEndpoint mcast;
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(addr),
                    mcast,
                    Wt * tile_bytes,
                    receivers,
                    {},
                    {.noc_x_start = x, .noc_y_start = y0, .noc_x_end = x, .noc_y_end = y1, .addr = addr});
                noc.async_write_barrier();
                for (uint32_t y = y0; y <= y1; ++y) {
                    ready.up(noc, x, y, 1);
                }
                noc.async_atomic_barrier();
            }
        }
        gamma.push_back(Wt);
    }
    {
        FUSED_ZONE("fz_gr_slab_rd_rest");
        read_run(noc, res_args, res_addr, C_RES, res_first + Wt, res_count - Wt);
    }
}
