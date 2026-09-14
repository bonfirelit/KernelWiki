---
id: hw-wmma-rdna4
title: "WMMA on RDNA4 (gfx12)"
type: hardware
architectures: [gfx1201, gfx1200, rdna4, rdna3]
tags: [wmma, wave32, swmmac]
confidence: source-reported
related: [hw-gfx1201, hw-mfma-cdna, hw-lds, migration-cdna-to-rdna4, kernel-rdna4-wmma-gemm]
sources: [doc-amd-rdna4-matrix-cores, blog-rdna4-wmma-lane-mapping]
aliases: [WMMA, "wave matrix multiply accumulate", v_wmma, "RDNA4 matrix core"]
---

# WMMA on RDNA4 (gfx12)

RDNA4's matrix core is reached through WMMA intrinsics, not MFMA. The operation
is `D := A*B + C` over a fixed **16x16** tile, issued cooperatively by all 32
lanes of a wave32 wavefront.

## Programming contract

The intrinsic name encodes everything:

```
__builtin_amdgcn_wmma_<C,D format>_16x16x16_<A,B format>_w32_gfx12
```

The `_gfx12` postfix is new — RDNA3/gfx11 intrinsics do not carry it, and the
two generations are **not** register-layout compatible (`doc-amd-rdna4-matrix-cores`).

```cpp
// gfx1201, wave32. Every lane in the wave must reach this call.
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

__global__ void wmma_16x16x16(const _Float16* A, const _Float16* B, float* D) {
  const int lane      = threadIdx.x;          // 0..31
  const int col       = lane % 16;            // fast-varying: N, not M
  const int row_base  = (lane / 16) * 8;      // 0 or 8

  half8  a_frag, b_frag;
  float8 c_frag = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int j = 0; j < 8; ++j) {
    a_frag[j] = A[(row_base + j) * 16 + col];  // A is column major (transposed)
    b_frag[j] = B[(row_base + j) * 16 + col];  // B is row major
  }

  c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, c_frag);

  for (int j = 0; j < 8; ++j)
    D[(row_base + j) * 16 + col] = c_frag[j];
}
```

## Fragment layout, and the trap

Each lane holds **8 elements** of each matrix: 8 x 32 lanes = 256 = 16 x 16
(`doc-amd-rdna4-matrix-cores`). For the accumulator, `blog-rdna4-wmma-lane-mapping`
derives the mapping explicitly:

```
VGPR[lane][j]         = matrix[(lane / 16) * 8 + j][lane % 16]
matrix[row][col]      -> VGPR[(row / 8) * 16 + col][row % 8]
```

The lane index selects the **column** (N); the accumulator register index
selects the row (M). Lanes 0-15 cover columns 0-15 / rows 0-7; lanes 16-31 cover
columns 0-15 / rows 8-15.

This is the layout bug that does not crash. Writing `row = lane % 16` compiles,
runs, and produces a silently transposed output tile. Verify against a reference
GEMM on a non-symmetric matrix before trusting any tiling built on top.

Orientation differs per operand: "B, C, and D matrices are row major while A is
transposed thus column major" (`doc-amd-rdna4-matrix-cores`).

## What changed from RDNA3

- RDNA3 duplicated some A and B elements across lanes; RDNA4 does not.
- RDNA3 split C/D into even/odd halves over the lower and upper 16 lanes, so
  chaining one WMMA's output into the next operand required a lane shuffle.
  RDNA4 removes that requirement.
- Lanes hold 8 elements on RDNA4 rather than 16 on RDNA3.

Chaining D into a subsequent B still needs a downcast, since D is f32 while C is
f16; `__builtin_amdgcn_cvt_pkrtz` packs two f32 into one VGPR.

## Boundary

- 16x16 only. Smaller problems are padded; larger ones are decomposed.
- `blog-rdna4-wmma-lane-mapping` documents the **accumulator** fragment; it
  publishes no separate A/B input-fragment table and makes no sparsity claim.
- The `iu4` integer form is documented
  (`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12`). No captured source in
  this wiki demonstrates a native FP4 WMMA input type on gfx1201 — see
  `hw-amd-narrow-precision`.
