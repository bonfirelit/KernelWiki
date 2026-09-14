---
id: technique-in-register-transpose
title: "In-Register Transpose for WMMA Operands"
type: technique
architectures: [gfx1201, gfx1200, rdna4, gfx950, cdna4]
tags: [in-register-transpose, wmma, ds-transpose, lds, wave32]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-wmma-rdna4, hw-lds]
related: [hw-wmma-rdna4, hw-lds, hw-mfma-cdna, technique-lds-bank-conflict-avoidance, kernel-rdna4-wmma-gemm]
sources: [doc-amd-rdna4-matrix-cores, blog-rdna4-wmma-lane-mapping, blog-rocm-mxfp4-rotation]
symptoms: [memory-bound, lds-bank-conflicts]
---

# In-Register Transpose for WMMA Operands

WMMA wants A column-major and B row-major (`doc-amd-rdna4-matrix-cores`). Real
tensors rarely arrive that way, so somewhere between global memory and the
matrix core a transpose has to happen. On CDNA4 the hardware does it. On RDNA4
it does not, and choosing where to put it is a real decision.

## Where the transpose can live

| Placement | RDNA4 (gfx1201) | CDNA4 (gfx950) |
|---|---|---|
| Hardware LDS transpose read | not available | `ds_read_tr16_b64` |
| LDS write pattern (scatter on store) | yes — costs a swizzle | yes |
| In registers via lane shuffle | yes | yes |
| Separate transpose kernel | yes — costs an HBM round trip | yes |

CDNA4's `ds_read_tr16_b64` "reads and transposes a 16-element tile from LDS in a
single operation", feeding MFMA the correct layout with no separate step
(`blog-rocm-mxfp4-rotation`). That instruction has no gfx12 counterpart.

## Transposing on the LDS store

Usually the cheapest option on RDNA4: you are already paying for the
`ds_write_b128`, so permute the index there and read back linearly. The
constraint is that the scattered write must stay bank-conflict-free, which is
exactly what `technique-lds-bank-conflict-avoidance` is for.

```cpp
// gfx1201, wave32. Stage a row-major 16x16 f16 tile into LDS transposed, so the
// subsequent fragment load reads A in the column-major order WMMA expects.
__global__ void stage_A_transposed(const _Float16* __restrict__ src,
                                   _Float16* __restrict__ frag_out) {
  __shared__ _Float16 tile[16][16 + 1];   // +1 breaks the bank alias on the
                                          // column-wise read below
  const int lane = threadIdx.x;           // 0..31
  const int col  = lane % 16;
  const int rb   = (lane / 16) * 8;

  // Scatter on store: element [r][c] of the source lands at tile[c][r].
  for (int j = 0; j < 8; ++j) {
    const int r = rb + j;
    tile[col][r] = src[r * 16 + col];
  }
  __syncthreads();

  // Linear read in WMMA fragment order: lane owns column `col`, rows rb..rb+7.
  for (int j = 0; j < 8; ++j)
    frag_out[(rb + j) * 16 + col] = tile[rb + j][col];
}
```

## Transposing in registers

Only when LDS is the binding resource. The constraint that makes this awkward on
RDNA4 is that the WMMA fragment layout and the lane-shuffle instructions both
take **constant** indices, so a general permutation becomes a fixed sequence of
exchanges, several of which move data a given lane does not need. Budget for
redundant traffic, and unroll fully so every `__shfl` index is a compile-time
constant.

## Verify, do not assume

A transpose bug does not crash — it produces a transposed result tile, which on
a symmetric test matrix looks correct. Test against a non-symmetric reference.
The lane mapping to check against is in `hw-wmma-rdna4`; that mapping was
compiled and run on this host against a CPU reference with 0 of 256 elements
mismatched, so it is a sound oracle.

## Transfers to CDNA?

Inverted: on CDNA4 prefer `ds_read_tr16_b64` and skip this technique entirely.
The LDS-store-transpose pattern still works there, it is just strictly more
work than the instruction the hardware provides.
