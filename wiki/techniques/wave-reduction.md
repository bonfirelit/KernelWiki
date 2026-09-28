---
id: technique-wave-reduction
title: "Wave-Level Reduction (DPP rows + ds_bpermute + readfirstlane)"
type: technique
architectures: [gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4]
tags: [wave-reduction, cross-lane, reduction, wave64, wave32]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-cross-lane]
related: [hw-cross-lane, technique-in-register-transpose, pattern-memory-bound]
sources: [blog-rocm-kernel-wiki, doc-cdna3-isa, doc-cdna4-isa, blog-amdgpu-kernel-opt-guide, doc-llvm-amdgpu-usage]
symptoms: [memory-bound]
---

# Wave-Level Reduction

Collapsing one value per lane into a scalar across the whole wavefront is the
inner step of every norm/softmax/attention-statistics kernel and of split-K
partials. CDNA has no single-instruction wave reduce; the canonical tree is
built from three cross-lane primitives in increasing cost (`hw-cross-lane`):

1. **DPP `row_shr` chain** within each 16-lane row — stays in the VALU, no
   LDS traffic. Four steps (shift 1/2/4/8) leave the row sum on the row's
   *last* lane (15/31/47/63); `bound_ctrl=true` fills vacated sources with
   the additive identity.
2. **`ds_bpermute`** to gather the four row results across the crossbar.
   Byte-address index (`lane << 2`), wraps mod 64 — see the traps on
   `hw-cross-lane`.
3. **`v_readfirstlane`** to uniformize the total into an SGPR.

```cpp
// Compiles for gfx950, gfx942, gfx1201 with ROCm 7.2.4 hipcc (compile-only;
// this host has no CDNA device). The wave64 tree; see below for wave32.
__device__ float wave_sum_shfl(float v) {      // portable baseline
  #pragma unroll
  for (int mask = 32; mask > 0; mask >>= 1)
    v += __shfl_xor(v, mask, 64);
  return v;
}

__device__ float wave_sum_dpp(float v) {
  int n = __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x111, 0xf, 0xf, true);
  v += __builtin_bit_cast(float, n);
  n = __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x112, 0xf, 0xf, true);
  v += __builtin_bit_cast(float, n);
  n = __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x114, 0xf, 0xf, true);
  v += __builtin_bit_cast(float, n);
  n = __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x118, 0xf, 0xf, true);
  v += __builtin_bit_cast(float, n);            // sums on lanes 15/31/47/63

  int bits = __builtin_bit_cast(int, v);
  float acc = __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(15 << 2, bits))
            + __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(31 << 2, bits))
            + __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(47 << 2, bits))
            + __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(63 << 2, bits));
  return __builtin_bit_cast(float,
      __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, acc)));
}

// gfx950 alternative for the cross-row steps: v_permlane16/32_swap keeps the
// exchange in the VALU, off the LDS unit. Precondition: the row partial must
// be replicated to every lane of its row (e.g. via row_bcast:15) -- the raw
// DPP chain leaves it on the row's last lane only.
__device__ float wave_sum_gfx950(float row_partial_replicated) {
#if defined(__gfx950__)
  int lane = __lane_id();
  int bits = __builtin_bit_cast(int, row_partial_replicated);
  auto p16 = __builtin_amdgcn_permlane16_swap(bits, bits, false, false);
  float v = row_partial_replicated
          + __builtin_bit_cast(float, (lane & 16) ? p16[0] : p16[1]);
  bits = __builtin_bit_cast(int, v);
  auto p32 = __builtin_amdgcn_permlane32_swap(bits, bits, false, false);
  return v + __builtin_bit_cast(float, (lane & 32) ? p32[0] : p32[1]);
#else
  return row_partial_replicated;
#endif
}
```

## Does it compute the same thing?

Yes — checked on silicon. `blog-rocm-kernel-wiki` ran a one-wave kernel on
MI350X/MI355X initializing lane `i` to `i + 1` and compared the explicit
DPP+`ds_bpermute` tree against the six-step `__shfl_xor` baseline: every lane
returned **2080** (the sum 1..64) from both paths. The explicit tree is not
faster by construction — it exists so you can hand-schedule it, and so the
gather count is visible.

## Why not reduce through LDS?

The naive tree writes 64 values to LDS, barriers, and reads them back. The
cross-lane tree uses **no LDS storage** and needs **no barrier** —
`ds_bpermute` borrows the crossbar but not a byte of capacity, and lanes of
one wavefront are implicitly in step between VALU ops. That makes it the
only reduction shape that composes inside a fused kernel where partials
already live in registers. For a *block* reduction across waves you still
pay one LDS round-trip: reduce within each wave first, then one scalar per
wave through LDS (`hw-lds`).

Payload is 32 bits per op: reduce `double`/`int64` as two halves, or reduce
in f32 and cast. For max/min, seed inactive lanes with the identity instead
of relying on `bound_ctrl`'s zero.

## Transfers to RDNA4?

**Yes, one level shorter.** gfx1201 is wave32: two 16-lane DPP rows per wave,
then a single `ds_bpermute` gather of lanes 15 and 31 (or one native
`v_permlane16` selector-form exchange) finishes the tree — the full wave64
snippet above compiles for gfx1201 but reduces the wrong grouping there, so
re-derive the indices. The building blocks all exist on RDNA4: DPP,
`__builtin_amdgcn_ds_bpermute`, `v_readfirstlane`, and — unlike CDNA — the
selector-form `v_permlane16` family (`hw-cross-lane` has the exact toolchain
split). What does not exist is the gfx950 `_swap` form, so the
`wave_sum_gfx950` path compiles to its fallback on gfx1201. As on CDNA, no
barrier is needed within the wave.
