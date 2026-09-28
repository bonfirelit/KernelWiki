---
id: lang-rocwmma
title: "rocWMMA"
type: language
tags: [rocwmma, mfma, wmma, gemm]
related: [lang-hip, lang-composable-kernel, hw-wmma-rdna4, hw-mfma-cdna]
sources: [doc-cdna3-isa, doc-cdna4-isa, blog-rocm-kernel-wiki]
reproducibility: snippet
architectures: [gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4]
confidence: source-reported
aliases: [rocWMMA, "rocWMMA fragment API"]
---

# rocWMMA

rocWMMA is AMD's header-only C++ fragment API over the matrix cores,
deliberately shaped like NVIDIA's `nvcuda::wmma`: declare opaque `fragment`
objects, fill them with `load_matrix_sync`, multiply with `mma_sync`, write
back with `store_matrix_sync`. The library picks the hardware instruction for
the target — `v_mfma_*` on CDNA (gfx942/gfx950, `hw-mfma-cdna`) and `v_wmma_*`
on RDNA (gfx1100/gfx1201, `hw-wmma-rdna4`) — so one source compiles across
MI300X, MI355X, and Radeon with no instruction-specific code.

The trade: the fragment's per-lane register layout is opaque. You get portable,
readable matrix-core code; you give up the register-level control that raw
builtins expose.

## Minimal fragment GEMM

Compile-checked on this host with `hipcc -O2 --offload-arch=gfx1201 -c`
(headers ship at `/opt/rocm/include/rocwmma`, ROCm 7.2.4). Each `mma_sync`
lowers to exactly one `v_wmma_f32_16x16x16_f16` in the emitted gfx1201 ISA.

```cpp
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

constexpr int M = 16, N = 16, K = 16;

// One wavefront computes one 16x16 output tile: C += A * B
__global__ void wmma_tile_gemm(const __half* A, const __half* B, float* C,
                               int lda, int ldb, int ldc, int Kdim)
{
    fragment<matrix_a,    M, N, K, __half, row_major> fragA;
    fragment<matrix_b,    M, N, K, __half, col_major> fragB;
    fragment<accumulator, M, N, K, float>             fragAcc;

    fill_fragment(fragAcc, 0.0f);

    for (int k = 0; k < Kdim; k += K) {
        load_matrix_sync(fragA, A + k, lda);
        load_matrix_sync(fragB, B + k, ldb);
        mma_sync(fragAcc, fragA, fragB, fragAcc);
    }

    store_matrix_sync(C, fragAcc, ldc, mem_row_major);
}
```

Every call is collective across the wavefront — 64 lanes on CDNA, 32 (or 64)
on RDNA — so launch whole waves and never hardcode a warp width.

## Porting from `nvcuda::wmma`

The call sequence matches name-for-name; the semantics do not:

| CUDA `nvcuda::wmma` | rocWMMA | What changes |
|---|---|---|
| `fragment<Use, M, N, K, T, Layout>` | same | layouts are opaque and per-architecture — re-derive any assumption |
| `load_matrix_sync` / `store_matrix_sync` | same | collective over 64 lanes on CDNA, not 32 |
| `mma_sync(d, a, b, c)` | same | accumulator lands in AGPRs on CDNA — it is a second register file with its own pressure |
| `warpSize == 32` | query it | 64 on CDNA, 32/64 on RDNA |

A port that treats fragments as black boxes usually drops in; a port that
peeked at fragment internals (lane mappings, element counts) does not —
gfx12 WMMA carries 8 elements per lane where RDNA3 carried a different count,
and the builtins behind the API are `_gfx12`-suffixed
(`blog-rocm-kernel-wiki`).

## When to use it, when not

- **Use rocWMMA** for fused epilogues, custom mixed-precision tiles, and
  CUDA-WMMA ports — anywhere portability and readability beat the last 10%.
- **Drop to raw `__builtin_amdgcn_mfma` / `__builtin_amdgcn_wmma`** when the
  schedule *is* the optimization: hand-placed `s_waitcnt`, explicit
  double-buffered LDS staging, scheduling barriers. That territory is
  `lang-hip`'s selection guidance and `technique-instruction-scheduling-amd`;
  a fragment API deliberately hides exactly those knobs.
- **Use `lang-composable-kernel`** when a full production pipeline already
  exists for your shape — CK's tuned GEMM/attention paths will beat a
  hand-written rocWMMA loop, because rocWMMA is a fragment layer, not an
  auto-tuner: the snippet above neither stages through LDS nor pipelines the
  matrix-core issue.

Gotchas: cross-lane shuffles on in-flight fragment data are undefined
(operate before load / after store); tile geometry is per-family, so
parameterize over `M/N/K` rather than assuming 16x16x16; and there is no
software fallback — a shape the silicon lacks does not compile.

## Transfers to RDNA4?

rocWMMA *is* the transfer story: the same fragment source compiles to WMMA on
gfx1201 (verified above). What does not transfer is performance intuition —
RDNA4 has no AGPR file, a smaller 64 KiB LDS, and its own WMMA tile geometry —
and any code that depended on MFMA fragment layouts. Treat a CDNA-tuned
rocWMMA kernel on RDNA4 as correct-but-untuned: re-derive occupancy with the
RDNA4 constants in `technique-occupancy-tuning-amd` before judging it.
