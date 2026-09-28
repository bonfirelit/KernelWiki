---
id: hw-cross-lane
title: "Cross-Lane Data Movement (DPP, ds_swizzle, ds_permute/bpermute, permlane)"
type: hardware
architectures: [gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4]
tags: [cross-lane, wave64, wave32, lds]
confidence: source-reported
related: [hw-lds, hw-amd-memory-ops, hw-gfx1201, technique-wave-reduction, technique-in-register-transpose]
sources: [blog-rocm-kernel-wiki, doc-cdna3-isa, doc-cdna4-isa, blog-amdgpu-kernel-opt-guide, doc-llvm-amdgpu-usage]
aliases: [DPP, "cross-lane", ds_bpermute, "ds_permute", permlane]
---

# Cross-Lane Data Movement

How one lane hands a VGPR to another lane without touching memory — the
substrate of wave reductions, broadcasts, and butterfly shuffles. CDNA is
wave64-only, so all of these operate over 64 lanes (`doc-cdna3-isa`); HIP's
`__shfl*` family lowers onto them. Take the cheapest rung that reaches your
pattern.

| Mechanism | Reach | Pattern | LDS unit? | Approx. cost |
|---|---|---|---|---|
| DPP modifier | within a 16-lane row | fixed in opcode | no | near-free, rides the VALU op |
| `ds_swizzle_b32` | 32-lane group | fixed, encoded mask | yes, no storage | ~50 cycles incl. wait |
| `ds_permute_b32` / `ds_bpermute_b32` | full wave64 | runtime per-lane index | yes, no storage | ~50 cycles incl. wait |
| `v_permlane16/32_swap_b32` | half-wave exchange, gfx950 only | none | no | VALU op |

The cycle figures are `blog-amdgpu-kernel-opt-guide`'s MI300 fused-softmax measurements; the `ds_*` rows include the `s_waitcnt`, DPP needs none.


## DPP: the free-ish rung

DPP is a modifier on an ordinary VALU op that swaps each lane's *operand* for
a neighbor's value before the ALU executes — the shuffle costs no extra issue
slot. Patterns view the wavefront as rows of 16 lanes: `row_shl`/`row_shr`/
`row_ror`, row broadcast, mirror (`doc-cdna3-isa`). Three things bite:

- **Result lane.** A `row_shr` reduction tree leaves the row sum on the row's
  *last* lane — 15/31/47/63 — not lane 0.
- **Vacated lanes.** With `bound_ctrl=true` a shifted-out source contributes
  0 (the additive identity), making a shift-based sum safe without
  predication; `false` preserves the old destination value.
- **EXEC hazard.** A VALU op does not forward its EXEC mask to a following
  DPP read, so the hardware requires wait states between writing a register
  and consuming it through DPP. The builtin inserts them; hand assembly must
  honor the documented wait-state count or read stale lane data
  (`doc-cdna3-isa`).

## The LDS-crossbar pair: ds_swizzle and ds_permute/ds_bpermute

`ds_swizzle_b32` moves one dword within 32-lane groups by an encoded XOR/
rotate pattern — the tool for fixed butterflies and quad transposes.
`ds_permute_b32` (push/scatter) and `ds_bpermute_b32` (pull/gather) move one
dword between *arbitrary* lanes under a runtime per-lane index
(`doc-cdna3-isa`). All three issue on the LDS unit but consume no LDS
storage: they contend with `ds_read`/`ds_write` bandwidth, not capacity, so
on an LDS-saturated kernel prefer a DPP or gfx950 permlane path even at
narrower reach.

`ds_bpermute`'s traps, each verified on MI350X/MI355X (`blog-rocm-kernel-wiki`):

- **Byte address.** The index is `lane << 2`; dropping the shift silently
  reads the wrong lane — no fault.
- **Wraps, not guards.** Source selection is `((addr + offset) / 4) % 64`:
  `64 << 2` reads lane 0. There is no out-of-range behavior to lean on.
- **EXEC-disabled source returns 0** — convenient for sums, treacherous for
  max/min.

## gfx950: permlane swaps stay in the VALU

CDNA4 adds `v_permlane16_swap_b32` / `v_permlane32_swap_b32`, exchanging the
paired 16-/32-lane halves of the wavefront with no LDS-unit round-trip
(`doc-cdna4-isa`). With the same value passed as both operands the partner is
result element 1 in the lower half, element 0 in the upper — neither element
is universally "self" (measured on MI355X, `blog-rocm-kernel-wiki`).

**Watch the name.** The RDNA selector form `__builtin_amdgcn_permlane16` /
`permlanex16` is a *different* instruction and does not assemble for gfx950 —
clang rejects it with "needs target feature gfx10-insts" (`blog-rocm-kernel-wiki`;
reproduced with this host's ROCm 7.2.4 hipcc: rejected for gfx950/gfx942,
accepted for gfx1201).

```cpp
// Compiles for gfx950/gfx942/gfx1201 with ROCm 7.2.4 hipcc (compile-only: no
// CDNA device on this host). Wave64 f32 sum.
__device__ float wave64_sum(float v) {
  // dpp_ctrl 0x111/0x112/0x114/0x118 == row_shr:1/2/4/8.
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
```

`v_readfirstlane_b32` (lowest active lane) and `v_readlane_b32` (named lane)
move one lane's value into an SGPR — the usual final step of a reduction, and
how a per-wave scalar gets to drive scalar control flow
(`doc-llvm-amdgpu-usage`).

## Transfers to RDNA4?

Mostly yes, one row narrower. gfx1201 is wave32 (`hw-gfx1201`): DPP rows are
still 16 lanes, but a wave holds two of them, so a full-wave tree is a level
shorter and every wave64 index table must be re-derived — the snippet above
compiles for gfx1201 but computes a *different* reduction there. DPP and
`ds_bpermute` both exist on RDNA, and RDNA4 natively has the selector-form
`v_permlane16` family that CDNA lacks; `__builtin_amdgcn_permlane16_swap`
is rejected for gfx1201. Port a `_swap`-based kernel to the selector form,
not the other way around.
