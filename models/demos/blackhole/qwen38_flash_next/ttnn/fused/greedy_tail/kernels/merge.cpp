// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One core: the per-core (value, id) pairs of scan.cpp -> the local maximum (bf16 TILE [1,1,1,1]: lane (0,0) of a
// zero tile), the local argmax (uint32 [1,1,1]) and the packed fp32 row [value | float(id)] the resolve's all_gather takes.
// Cores are visited in increasing order (increasing id ranges), so the first strict maximum is the lowest id.
// Named compile-time args: cb_stage, cores.  Compile-time args: TensorAccessorArgs(pairs), (zero tile), (values),
// (indices), (packed).  Runtime args: 0 pairs addr, 1 zero-tile addr, 2 values addr, 3 indices addr, 4 packed addr.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_STAGE = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t CORES = get_named_compile_time_arg_val("cores");
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t PAIRS_BYTES = ((16 * CORES) + 63) & ~63u;

FORCE_INLINE uint32_t key_of(uint32_t fp32_bits) {
    fp32_bits = (fp32_bits & 0x7FFFFFFFu) ? fp32_bits : 0u;  // scan.cpp already canonicalized -0.0; keep the rule here
    return (fp32_bits & 0x80000000u) ? ~fp32_bits : (fp32_bits | 0x80000000u);
}

void kernel_main() {
    constexpr auto a_pairs = TensorAccessorArgs<0>();
    constexpr auto a_zero = TensorAccessorArgs<a_pairs.next_compile_time_args_offset()>();
    constexpr auto a_values = TensorAccessorArgs<a_zero.next_compile_time_args_offset()>();
    constexpr auto a_indices = TensorAccessorArgs<a_values.next_compile_time_args_offset()>();
    constexpr auto a_packed = TensorAccessorArgs<a_indices.next_compile_time_args_offset()>();
    const auto pairs = TensorAccessor(a_pairs, get_arg_val<uint32_t>(0));
    const auto zero = TensorAccessor(a_zero, get_arg_val<uint32_t>(1));
    const auto values = TensorAccessor(a_values, get_arg_val<uint32_t>(2));
    const auto indices = TensorAccessor(a_indices, get_arg_val<uint32_t>(3));
    const auto packed = TensorAccessor(a_packed, get_arg_val<uint32_t>(4));

    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    constexpr uint32_t STAGE_TILE = 0, STAGE_PAIRS = TILE_BYTES, STAGE_OUT = TILE_BYTES + PAIRS_BYTES;
    noc.async_read(zero, stage, TILE_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_TILE});
    noc.async_read(pairs, stage, PAIRS_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_PAIRS});
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    uint32_t best_key = 0, best_bits = 0, best_id = 0;
    for (uint32_t c = 0; c < CORES; ++c) {
        const uint32_t bits = words[STAGE_PAIRS / 4 + 4 * c];
        const uint32_t key = key_of(bits);
        if (c == 0 || key > best_key) {
            best_key = key;
            best_bits = bits;
            best_id = words[STAGE_PAIRS / 4 + 4 * c + 1];
        }
    }
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + STAGE_TILE);
    tile[0] = best_bits >> 16;  // lane (0, 0) of the value tile
    union {
        float f;
        uint32_t u;
    } id_fp32;
    id_fp32.f = static_cast<float>(best_id);  // the chain's typecast: exact below 2^24 (soft-float int -> fp32)
    words[STAGE_OUT / 4] = best_bits;
    words[STAGE_OUT / 4 + 1] = id_fp32.u;
    words[STAGE_OUT / 4 + 2] = 0;
    words[STAGE_OUT / 4 + 3] = 0;
    words[STAGE_OUT / 4 + 4] = best_id;
    noc.async_write(stage, values, TILE_BYTES, {.offset_bytes = STAGE_TILE}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(stage, packed, 8, {.offset_bytes = STAGE_OUT}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(stage, indices, 4, {.offset_bytes = STAGE_OUT + 16}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write_barrier();
    stage.push_back(1);
}
