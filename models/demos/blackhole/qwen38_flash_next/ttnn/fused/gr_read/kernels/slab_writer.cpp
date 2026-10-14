// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Writes whole tiles from one CB into a TILE tensor along reader.cpp's stream walk: pages
// first + o * outer_stride + i * inner_stride for o < outer, i < inner, popped `batch` at a time (the slab programs'
// row blocks: a branch-major row block lands on its flat pages 80r + 20b + j).
// Compile-time args: 0 cb, 1.. the tensor's TensorAccessorArgs.
// Runtime args: 7 (addr, outer, inner, first, inner_stride, outer_stride, batch).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t CB = get_compile_time_arg_val(0);
constexpr uint32_t ACCESSOR_BASE = 1;

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<ACCESSOR_BASE>();
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t outer = get_arg_val<uint32_t>(1);
    const uint32_t inner = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t inner_stride = get_arg_val<uint32_t>(4);
    const uint32_t outer_stride = get_arg_val<uint32_t>(5);
    const uint32_t batch = get_arg_val<uint32_t>(6);
    const auto accessor = TensorAccessor(args, addr);
    const uint32_t tile_bytes = get_tile_size(CB);
    Noc noc;
    DataflowBuffer dfb(CB);
    {
        FUSED_ZONE("fz_gr_slab_wr_main");
        uint32_t pending = 0;
        for (uint32_t o = 0; o < outer; ++o) {
            for (uint32_t i = 0; i < inner; ++i) {
                if (pending == 0) {
                    dfb.wait_front(batch);
                }
                noc.async_write(
                    dfb,
                    accessor,
                    tile_bytes,
                    {.offset_bytes = pending * tile_bytes},
                    {.page_id = first + o * outer_stride + i * inner_stride});
                if (++pending == batch) {
                    noc.async_writes_flushed();
                    dfb.pop_front(batch);
                    pending = 0;
                }
            }
        }
        noc.async_write_barrier();
    }
}
