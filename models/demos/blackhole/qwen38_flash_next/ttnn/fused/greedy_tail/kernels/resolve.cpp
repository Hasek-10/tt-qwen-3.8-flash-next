// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One core: the four devices' packed [value | local id] fp32 pairs (the all_gather of merge.cpp's row) -> the greedy
// token row: owner d's value lowered by owner_tie_break[d] (fp32 subtract, as the chain's SFPU fp32 subtract), the
// first maximum in owner order (ttnn.argmax's lowest index), the owner's id plus lm_head_vocab_starts[owner] (fp32
// add, exact below 2^24), written as lane 0 of row 0 of an fp32 TILE [1,1,1,32] whose other lanes are 0.0 (the
// chain's unit_column multiply).  The RISC's soft-float subtract and add are IEEE round-to-nearest-even, as the SFPU's.
// Named compile-time args: cb_stage, devices.  Compile-time args: TensorAccessorArgs(gathered), (tie_break),
// (vocab_starts), (zero fp32 tile), (token_row).  Runtime args: the five buffer addresses in that order.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_STAGE = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t DEVICES = get_named_compile_time_arg_val("devices");
constexpr uint32_t FP32_TILE_BYTES = 4096;
constexpr uint32_t GRAIN = 64;

void kernel_main() {
    constexpr auto a_gathered = TensorAccessorArgs<0>();
    constexpr auto a_tie = TensorAccessorArgs<a_gathered.next_compile_time_args_offset()>();
    constexpr auto a_starts = TensorAccessorArgs<a_tie.next_compile_time_args_offset()>();
    constexpr auto a_zero = TensorAccessorArgs<a_starts.next_compile_time_args_offset()>();
    constexpr auto a_token = TensorAccessorArgs<a_zero.next_compile_time_args_offset()>();
    const auto gathered = TensorAccessor(a_gathered, get_arg_val<uint32_t>(0));
    const auto tie = TensorAccessor(a_tie, get_arg_val<uint32_t>(1));
    const auto starts = TensorAccessor(a_starts, get_arg_val<uint32_t>(2));
    const auto zero = TensorAccessor(a_zero, get_arg_val<uint32_t>(3));
    const auto token = TensorAccessor(a_token, get_arg_val<uint32_t>(4));

    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    constexpr uint32_t STAGE_TILE = 0, STAGE_GATHERED = FP32_TILE_BYTES, STAGE_TIE = STAGE_GATHERED + GRAIN, STAGE_STARTS = STAGE_TIE + GRAIN;
    noc.async_read(zero, stage, FP32_TILE_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_TILE});
    noc.async_read(gathered, stage, GRAIN, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_GATHERED});
    noc.async_read(tie, stage, GRAIN, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_TIE});
    noc.async_read(starts, stage, GRAIN, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_STARTS});
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr float* g = reinterpret_cast<volatile tt_l1_ptr float*>(base + STAGE_GATHERED);
    volatile tt_l1_ptr float* t = reinterpret_cast<volatile tt_l1_ptr float*>(base + STAGE_TIE);
    volatile tt_l1_ptr float* s = reinterpret_cast<volatile tt_l1_ptr float*>(base + STAGE_STARTS);
    uint32_t owner = 0;
    float best = g[0] - t[0];
    for (uint32_t d = 1; d < DEVICES; ++d) {
        const float ranked = g[2 * d] - t[d];
        if (ranked > best) {
            best = ranked;
            owner = d;
        }
    }
    const float id = g[2 * owner + 1] + s[owner];
    volatile tt_l1_ptr float* row = reinterpret_cast<volatile tt_l1_ptr float*>(base + STAGE_TILE);
    row[0] = id;  // lane (0, 0) of the token tile; the rest of the zero tile stays 0.0
    noc.async_write(stage, token, FP32_TILE_BYTES, {.offset_bytes = STAGE_TILE}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write_barrier();
    stage.push_back(1);
}
