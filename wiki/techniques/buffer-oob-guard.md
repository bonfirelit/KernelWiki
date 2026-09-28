---
id: technique-buffer-oob-guard
title: "Branchless Bounds Checking with Buffer Descriptors"
type: technique
architectures: [gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4]
tags: [bounds-checking, vectorized-loads, gemm, attention]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-amd-memory-ops]
related: [hw-amd-memory-ops, hw-lds, technique-vectorized-loads]
sources: [blog-rocm-kernel-wiki, doc-cdna3-isa, doc-cdna4-isa, blog-amdgpu-kernel-opt-guide, doc-llvm-amdgpu-usage]
symptoms: [memory-bound]
---

# Branchless Bounds Checking with Buffer Descriptors

The tail-tile problem: a GEMM with `M=4097` and 256-row tiles has a one-row
final tile, and the textbook fix — a per-lane `if (row < M)` — diverges the
wavefront and still computes the out-of-bounds address on the masked lanes.
MUBUF buffer instructions offer a hardware alternative: size the descriptor
honestly and let the memory system do the check, branchlessly.

## The contract

A `buffer_load`/`buffer_store` addresses memory through a 128-bit resource
descriptor carrying base, stride, and `num_records`. For a raw buffer the
hardware compares `voffset + inst_offset` against `num_records` (a byte
count) on every access: **out-of-bounds loads return 0, out-of-bounds stores
are dropped** (`doc-cdna3-isa`, `doc-cdna4-isa`). No predicate code, no EXEC
divergence, no fault — verified on MI355X with a 32-byte window: lanes 8–63
of a 64-lane probe read exactly 0 (`blog-rocm-kernel-wiki`). Contrast with
`global_load`/`flat`, which take a raw pointer and have no object-size guard
(`hw-amd-memory-ops`).

Details that are not optional:

- **Vector accesses check per dword component.** A `dwordx4` that straddles
  `num_records` returns its in-range dwords followed by zeros — the valid
  prefix of a tail tile survives without scalarizing
  (`blog-rocm-kernel-wiki`, verified on MI355X with a `b128` load starting at
  byte 24 of a 32-byte window).
- **`soffset` is outside the check.** The SGPR offset moves the final
  address but not the comparison; keep `soffset=0` and put the whole
  window-relative displacement in `voffset`.
- **Flags word 3 must be initialized.** `0x00020000` on gfx9/gfx942/gfx950,
  `0x31004000` on gfx1201. A zero flags word marks *every* lane out-of-bounds
  — the kernel runs fine and returns all zeros (`blog-rocm-kernel-wiki`).
- **The window is ~4 GiB.** `num_records` is a 32-bit byte extent. Larger
  tensors are not blocked: rebase the descriptor (`base += window_start`,
  `offset -= window_start`) per chunk, and never compute
  `n_elems * sizeof(T)` in a signed 32-bit int on the way.

```cpp
// Compiles for gfx950 and gfx1201 (and gfx942) with ROCm 7.2.4 hipcc.
// flags is descriptor word 3: 0x00020000 on gfx942/gfx950, 0x31004000 on
// gfx1201 -- left as a parameter so the call site owns the target split.
__device__ float guarded_load(const float* window_base, uint32_t byte_offset,
                              uint32_t valid_bytes, uint32_t flags) {
  __amdgpu_buffer_rsrc_t rsrc = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<float*>(window_base),
      /*stride=*/(short)0,          // raw byte buffer
      /*num_records=*/valid_bytes,
      /*flags=*/flags);
  int raw = __builtin_amdgcn_raw_buffer_load_b32(
      rsrc, /*voffset=*/byte_offset, /*soffset=*/0, /*aux=*/0);
  return __builtin_bit_cast(float, raw);  // lanes at/past valid_bytes -> 0.0f
}
```

## When zero is the wrong identity

OOB-returns-0 is free only when 0 is your neutral element — sums, dot
products, GEMM accumulator tiles. It actively lies for:

- **max/min reductions** (the FlashAttention running max): OOB lanes
  contribute 0.0f and can win the max. Seed the accumulator with `-inf`, or
  mask after the load.
- **Accumulators read back later:** a dropped OOB store leaves stale bytes;
  anything that will be re-read must have been initialized.
- **Division** by a guarded-loaded value: 0 in the denominator gives inf.
  Add the bias first.

## Where it pays

GEMM and attention tail tiles without a second, branchy cleanup loop — one
code path serves the interior and the ragged edge. It also removes the
per-iteration compare from the hot loop and shrinks the instruction
footprint. Pairs naturally with vectorized loads (`technique-vectorized-loads`)
since the per-component check keeps wide accesses alive into the tail, and
with direct-to-LDS prefetch on CDNA, where a full-tile guard would serialize
the async copy behind a predicate (`blog-rocm-kernel-opt-guide`,
`hw-amd-memory-ops`). If a compiler pass demotes your `buffer_load` to a
`global_load` (descriptor not provably loop-invariant), the guard is silently
gone — check the emitted ISA for `buffer_` vs `global_`.

## Transfers to RDNA4?

**Yes, with a different flags word.** The descriptor mechanism, the
OOB-returns-0/dropped-store contract, and the `__builtin_amdgcn_raw_buffer_load`
family all exist on gfx1201 — the snippet above compiles for it. What changes
is word 3 of the descriptor: `0x31004000`, not `0x00020000`
(`blog-rocm-kernel-wiki`). RDNA4 has no direct-to-LDS path, so the
prefetch-pairing benefit applies to the VGPR staging route instead.
