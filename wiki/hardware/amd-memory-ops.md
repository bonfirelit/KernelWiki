---
id: hw-amd-memory-ops
title: "AMD global memory ops, direct-to-LDS, and wait counters"
type: hardware
architectures: [gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4]
tags: [global-load-lds, lds, vectorized-loads, double-buffering]
confidence: source-reported
related: [hw-lds, hw-gfx1201, hw-mfma-cdna, technique-double-buffering, technique-vectorized-loads, technique-buffer-oob-guard, kernel-cdna4-fp8-gemm]
sources: [doc-rocm-workload-optimization, doc-llvm-amdgpu-usage, blog-rocm-fp8-gemm-cdna4, blog-rocm-memory-scheduling, blog-rocm-kernel-wiki, doc-cdna3-isa, doc-cdna4-isa, blog-amdgpu-kernel-opt-guide]
aliases: [global_load_lds, buffer_load_lds, "direct-to-LDS", vmcnt, lgkmcnt, s_waitcnt]
---

# AMD global memory ops, direct-to-LDS, and wait counters

The AMD counterpart to the question "how do I get a tile from HBM into fast
memory without stalling". There is no TMA and no descriptor object; the moving
parts are the load instruction width, an optional direct-to-LDS path, and two
hardware counters you drain by hand.

## Programming contract

```cpp
// Get the widest load the ISA offers. The ROCm workload guide's ISA checklist
// is literally "confirm global_load_dwordx4 is used" --- a scalar float load
// per lane leaves 4x of the request width on the floor.
__global__ void copy_wide(const float4* __restrict__ src, float4* __restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];   // -> global_load_dwordx4 / global_store_dwordx4
}
```

## The two counters

- **`vmcnt`** — outstanding vector-memory (global / buffer) operations.
- **`lgkmcnt`** — outstanding LDS, GDS, constant, and message traffic.

`s_waitcnt vmcnt(n)` and `s_waitcnt lgkmcnt(n)` block until at most `n` are
still in flight, so the *non-zero* forms are the whole point: a software
pipeline issues N loads and then waits on `vmcnt(N-1)` to consume the first
while the rest remain outstanding. Waiting on `(0)` everywhere is correct and
slow. Reading these operands out of the ISA dump is the documented way to see
whether the compiler pipelined your loop (`doc-rocm-workload-optimization`).

## How deep the counters go

The counters are small saturating fields, and their widths bound pipeline
depth (`doc-cdna3-isa`, `doc-cdna4-isa`):

- **`vmcnt` is 6-bit** — up to 63 outstanding VMEM ops per wave, enough for
  deep load pipelines.
- **`lgkmcnt` is 4-bit** — and LDS, GDS, scalar-constant, and message traffic
  all share it, so a wave interleaving `s_load_*` with `ds_read_*` must
  reason about both when picking a wait target.
- **`expcnt` is 3-bit** and covers export/GDS; the CDNA4 ISA documents it as
  unused, and compute kernels never emit an `expcnt` wait.

The load-bearing rule is about completion order: **same-type memory ops
complete in issue order.** Two `global_load`s retire in the order they were
issued, so waiting `vmcnt(N)` deterministically means "all but the last N
issued VMEM ops have landed" — that is what makes the non-zero waits in a
software pipeline exact rather than hopeful. Different-type ops have **no**
relative completion order: a `ds_read` (`lgkmcnt`) and a `global_load`
(`vmcnt`) may retire in either order, so wait on each counter you depend on.
Counters are per-wave; cross-wave visibility still needs `s_barrier`
(`__syncthreads`).

## The three instruction classes

Every vector memory op is one of three classes, and they differ in exactly
the ways a pipeline cares about — how the address is formed, what
out-of-bounds means, and which counter the op charges (`doc-cdna3-isa`,
`doc-cdna4-isa`):

| Class | Address model | Bounds net | Counters |
|---|---|---|---|
| `global_*` | 64-bit pointer, or SGPR base + 32-bit VGPR offset | none — guard in software | `vmcnt` |
| `buffer_*` (MUBUF) | 128-bit V# resource descriptor + VGPR offset | hardware: OOB load reads 0, OOB store drops | `vmcnt` |
| `flat_*` | one 64-bit address, aperture resolved at runtime | none | **both** `vmcnt` and `lgkmcnt` |

The picks in practice: `global` for plain HIP pointer arithmetic with
explicit guards, `buffer` when you want the bounds check for free, and `flat`
only when the address space is genuinely unknowable at compile time. Because
a `flat` op can land in global *or* LDS it charges both counters, and the two
retire out of order — after dependent flat traffic, `s_waitcnt 0` (every
counter drained) is the only fully safe wait. That is one more reason
performance kernels keep `flat` out of the hot loop.

## The V# descriptor and the free bounds net

A MUBUF op takes no pointer; it takes a 128-bit **resource descriptor** (V#)
in four consecutive SGPRs packing base address, stride, `num_records`, and a
flags word — a register descriptor, still nothing like TMA's memory-resident
object. For a raw byte buffer the hardware compares
`voffset + inst_offset` against `num_records` on every access, and an
out-of-bounds lane **loads 0 / drops its store**, branchlessly — no fault, no
EXEC divergence (`doc-cdna3-isa`; silicon-verified on MI355X by
`blog-rocm-kernel-wiki` with a 32-byte-window probe). The exploitation side —
tail tiles, and the cases where zero is the wrong identity — lives in
`technique-buffer-oob-guard`; the hardware facts that make it work:

- **The check is per dword component** for `dwordx2`/`x3`/`x4`: a vector that
  straddles `num_records` returns its in-range dwords and zeros only the rest
  (verified on MI355X with a `b128` load crossing the window edge,
  `blog-rocm-kernel-wiki`).
- **`soffset` is outside the comparison** — it moves the final address only.
  Keep `soffset=0` and put the whole window-relative displacement in the VGPR
  offset, or the guard silently covers the wrong range.
- **Flags word 3 must be initialized**: `0x00020000` on gfx9 (gfx942/gfx950),
  `0x31004000` on gfx1201. A zero flags word marks *every* lane OOB — the
  kernel runs fine and returns all zeros (`blog-rocm-kernel-wiki`).
- **The raw window is ~4 GiB.** `num_records` is a 32-bit byte extent
  (`blog-amdgpu-kernel-opt-guide`). Larger tensors are not blocked — rebase
  the descriptor's 64-bit base per chunk and keep offsets window-local — but
  never truncate `size_t` arithmetic into the descriptor.

```cpp
// Compiles for gfx942, gfx950, and gfx1201 with ROCm 7.2.4 hipcc.
#include <hip/hip_runtime.h>
#include <stdint.h>

#if !defined(__HIP_DEVICE_COMPILE__) || !__HIP_DEVICE_COMPILE__
#define WIKI_RAW_BUFFER_FLAGS 0xffffffffu  // host pass; never issued
#elif defined(__gfx942__) || defined(__gfx950__)
#define WIKI_RAW_BUFFER_FLAGS 0x00020000u  // gfx9 raw-buffer flags word
#elif defined(__gfx1201__)
#define WIKI_RAW_BUFFER_FLAGS 0x31004000u  // RDNA4 raw-buffer flags word
#else
#error "Add the target's raw-buffer flags before using this helper"
#endif

using f32v4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

// Bounds-checked 16-byte load through a raw descriptor; OOB dwords read 0.
__device__ f32v4 load_tile_guarded(const float* window_base,
                                   uint32_t valid_bytes,
                                   uint32_t byte_off) {
  __amdgpu_buffer_rsrc_t rsrc = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<float*>(window_base),
      /*stride=*/(short)0,              // raw byte buffer
      /*num_records=*/valid_bytes,
      /*flags=*/WIKI_RAW_BUFFER_FLAGS);
  return __builtin_amdgcn_raw_buffer_load_b128(
      rsrc, byte_off, /*soffset=*/0, /*aux=*/0);
}

__global__ void copy_guarded(const float* base, float* out,
                             uint32_t valid_bytes, uint32_t tile_off) {
  f32v4 v = load_tile_guarded(base, valid_bytes, tile_off + 16 * threadIdx.x);
  __builtin_amdgcn_s_waitcnt(0);
  for (int j = 0; j < 4; ++j) out[4 * threadIdx.x + j] = v[j];
}

#undef WIKI_RAW_BUFFER_FLAGS
```

## Direct global-to-LDS (CDNA only)

CDNA can move data from global memory into LDS without landing it in VGPRs
first, reached through `llvm.amdgcn.raw.buffer.load.lds`
(`llvm_amdgcn_raw_buffer_load_lds` in HIP). CDNA4 widens the per-lane
`GLOBAL_LOAD_LDS` transfer to "Up to 128 bits/lane" against 32-bit on CDNA3.

The exact legal per-lane sizes for `__builtin_amdgcn_load_to_lds` are
**{1, 2, 4} bytes on gfx942** and **{1, 2, 4, 12, 16} bytes on gfx950** —
there is no 8-byte form on either generation (silicon-verified,
`blog-rocm-kernel-wiki`; on ROCm 7.2.4 clang-22 an 8-byte request fails with
"invalid size value" and the gfx950 12/16-byte forms lower to
`global_load_lds_dwordx3`/`dwordx4`). Two disciplines come with the path: the
LDS write address derives from the `M0` register (shared with `ds_*` indexed
ops — re-establish it if you interleave them), and completion is
**vmcnt-tracked even though the destination is LDS**, so you drain it like
any other global load, then `s_barrier` before other waves read the tile.

```cpp
// gfx942 / gfx950 only -- gfx1201 has no direct-to-LDS at all.
#include <hip/hip_runtime.h>

__global__ void prefetch_tile(const float* __restrict__ g, float* out) {
  __shared__ float tile[256];
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
  // Wave-uniform LDS base; hardware applies the per-lane offset. Size 4 is
  // legal on both CDNA targets; 12/16 are gfx950-only.
  float* dst = &tile[(threadIdx.x / 64) * 64];
  __builtin_amdgcn_load_to_lds(const_cast<float*>(g + threadIdx.x), dst,
                               /*size=*/4, /*offset=*/0, /*aux=*/0);
  __builtin_amdgcn_s_waitcnt(0);   // drain vmcnt before consuming LDS
  __syncthreads();
  out[threadIdx.x] = tile[threadIdx.x];
#else
  out[threadIdx.x] = 0.f;          // host pass placeholder
#endif
}
```

In `blog-rocm-fp8-gemm-cdna4`'s ladder this single change moved a 4096-cubed
FP8 GEMM from 336.88 to 506.70 TFLOP/s, and it is what makes the subsequent
double-buffering rung work: "the next tile's global-to-LDS transfer runs in
parallel with current MFMA work", synchronized with `s_waitcnt vmcnt(...)`.

**Transfers to RDNA4? No.** gfx1201 has no direct-to-LDS path at any width —
ROCm 7.2.4 rejects the builtin outright for gfx1201 ("needs target feature
vmem-to-lds-load-insts"). The buffer descriptor made the trip, but with a
different flags word: `0x31004000` on gfx1201 against gfx9's `0x00020000`
(`blog-rocm-kernel-wiki`). On RDNA4 the tile
takes the long route — `global_load_dwordx4` into VGPRs, then `ds_write_b128`
into LDS — and the VGPRs in flight count against occupancy for the duration.
That is a real cost, and it is one reason RDNA4 GEMM kernels favour smaller
K-tiles than their CDNA equivalents.

## Non-temporal wide ops

A compiler-native `float __attribute__((ext_vector_type(4)))` dereferenced
through `__builtin_nontemporal_load` / `__builtin_nontemporal_store` lowers
to `global_load_dwordx4 ... nt` / `global_store_dwordx4 ... nt` on both
gfx942 and gfx950 (compile + ISA checked on ROCm 7.2.4 clang-22). Use the
native vector type, not HIP's `float4` wrapper — older ROCm releases reject
the wrapper as the builtin's pointer type. And treat `nt` as a *lowering*
fact, not a cache-policy promise: which cache level the hint actually
bypasses is per-target. The use case is streaming tiles you will not
re-read, keeping them from evicting the working set.

```cpp
// Compiles for gfx942 and gfx950 with ROCm 7.2.4 hipcc.
#include <hip/hip_runtime.h>

typedef float f32v4 __attribute__((ext_vector_type(4)));

__global__ void copy_nt(const f32v4* __restrict__ src,
                        f32v4* __restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    __builtin_nontemporal_store(__builtin_nontemporal_load(src + i), dst + i);
}
```

## The full path

`blog-rocm-memory-scheduling` traces a three-stage pipelined CDNA GEMM end to
end: HBM -> VGPR -> LDS -> AGPR. Per K-tile each lane issues 16
`buffer_load_dwordx4` for 256 B, stages through a 64 KiB LDS region with
`ds_write_b128` / `ds_read_b128` pairs, and feeds MFMA out of AGPRs that never
spill. On RDNA4 the same diagram loses its last arrow: accumulators stay in
ordinary VGPRs.
