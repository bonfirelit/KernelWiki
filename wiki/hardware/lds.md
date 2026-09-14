---
id: hw-lds
title: "LDS (Local Data Share)"
type: hardware
architectures: [gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4]
tags: [lds, ds-transpose, lds-bank-conflict-avoidance, swizzling]
confidence: source-reported
related: [hw-mfma-cdna, hw-wmma-rdna4, hw-amd-memory-ops, technique-lds-bank-conflict-avoidance, pattern-lds-bank-conflicts]
sources: [doc-llvm-amdgpu-usage, doc-rocm-workload-optimization, blog-rocm-occupancy-mi355x, blog-rocm-fp8-gemm-cdna4, blog-rocm-memory-scheduling, blog-rocm-mxfp4-rotation]
aliases: [LDS, "local data share", "Local Data Share", "group segment"]
---

# LDS (Local Data Share)

AMD's on-chip scratchpad — the structural counterpart of CUDA shared memory,
but with its own instruction family and its own capacity story per generation.

## Programming contract

LDS is **address space 3** in the AMDGPU backend, and it is a *32-bit* space
while global is 64-bit (`doc-llvm-amdgpu-usage`). That is why LDS traffic uses
`ds_read` / `ds_write` rather than an addressing mode on `global_load`, and why
LDS accesses are counted by `lgkmcnt` while global accesses are counted by
`vmcnt`.

```cpp
// gfx1201: 64 KiB of LDS per workgroup. Ask for 128-bit accesses --- the
// ROCm workload guide's ISA checklist wants ds_read_b128 / ds_write_b128.
__global__ void stage_tile(const float4* __restrict__ g, float4* __restrict__ out) {
  __shared__ float4 tile[16][16 + 1];       // +1 float4 of padding: see below
  const int r = threadIdx.y, c = threadIdx.x;

  tile[r][c] = g[r * 16 + c];               // ds_write_b128
  __syncthreads();                           // s_barrier + s_waitcnt lgkmcnt(0)
  out[c * 16 + r] = tile[c][r];             // ds_read_b128, transposed
}
```

## Capacity is generation-specific

| Target | LDS per CU | Source |
|---|---|---|
| gfx1201 (RDNA4) | 64 KiB | measured on this host, `hw-gfx1201` |
| gfx942 (CDNA3) | 64 KiB | `doc-rocm-workload-optimization` uses `floor(65536 / L)` |
| gfx950 (CDNA4) | 160 KiB | `blog-rocm-occupancy-mi355x`; the guide uses `floor(163840 / L)` |

CDNA4's extra 96 KiB is *not* an invitation to move the accumulator off-chip.
`blog-rocm-occupancy-mi355x` is explicit that it is "a latency-hiding budget,
not an accumulator substitute" — spend it on deeper prefetch of the operand
stream.

## Bank conflicts

`ds_read_b128` is executed in four phases and **each phase must be
conflict-free** (`blog-rocm-fp8-gemm-cdna4`). The remedy that blog measures is a
row-based XOR remap on 16-byte columns, `swizzled_col = c ^ mask(r)` with
`mask(r) = perm << 4`, which is self-inverse. See
`technique-lds-bank-conflict-avoidance` for when to prefer XOR swizzle over
padding.

## Transpose reads

CDNA4 adds `ds_read_tr16_b64`, which "reads and transposes a 16-element tile
from LDS in a single operation" so MFMA gets the right operand layout with no
separate transpose pass (`blog-rocm-mxfp4-rotation`).

**Transfers to RDNA4? No.** gfx1201 has no LDS transpose read. On RDNA4 the
transpose has to happen either in the LDS write pattern or in registers — see
`technique-in-register-transpose`.

## Synchronization

`blog-rocm-memory-scheduling` describes the discipline a double-buffered GEMM
needs: two `s_barrier`s per iteration, the first ensuring every wave finished
*reading* the previous K-tile before the region is overwritten, the second
ensuring every wave finished *writing* before any wave reads. LDS writes are
drained with `s_waitcnt lgkmcnt(0)`; each MFMA waits on `lgkmcnt` so it cannot
consume an operand `ds_read_b128` has not returned.
