---
id: pattern-lds-bank-conflicts
title: "LDS Bank Conflicts"
type: pattern
tags: [lds, lds-bank-conflict-avoidance, swizzling, shared-memory-optimization]
symptoms: [lds-bank-conflicts, high-lgkmcnt-wait, memory-bound, low-compute-utilization]
candidate_techniques: [technique-lds-bank-conflict-avoidance, technique-swizzling, hw-lds, technique-in-register-transpose]
related: [pattern-memory-bound, pattern-pipeline-stalls, hw-lds, kernel-cdna4-fp8-gemm]
sources: [blog-rocm-fp8-gemm-cdna4, doc-rocm-workload-optimization, blog-rocm-memory-scheduling]
architectures: [gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4]
confidence: source-reported
---

# LDS Bank Conflicts

## Symptom

The kernel is not memory-bandwidth-bound at the HBM level, the matrix core is
idle a large fraction of the time, and the ISA shows the wave parked on
`s_waitcnt lgkmcnt(...)` rather than `vmcnt(...)`. The profiler reports a
non-trivial LDS bank-conflict rate. Widening LDS accesses from `_b32` to `_b128`
for bandwidth made things *worse* rather than better.

## Likely Causes

- **A power-of-two row stride.** A tile declared `float tile[N][32]` puts every
  column-wise access in the same bank. The classic case, and the reason
  `[N][32+1]` is such a common idiom.
- **A wide access that conflicts per-phase.** `ds_read_b128` executes in four
  phases and *each phase* must independently be conflict-free
  (`blog-rocm-fp8-gemm-cdna4`). A layout verified against `ds_read_b32` can
  conflict badly at `b128`. This is the cause that surprises people, because it
  appears when following the bandwidth advice.
- **A transposed read of a linearly written tile.** Storing row-major and reading
  column-major to feed the matrix core is the standard source; on gfx1201 there
  is no `ds_read_tr16_b64` to absorb it.
- **Padding lost to vectorization.** A `+1` of padding that was one `float`
  becomes meaningless once the access type widens to `float4`.

## Candidate Techniques

| Technique | When |
|---|---|
| `technique-lds-bank-conflict-avoidance` | First stop. Covers both padding and the XOR-swizzle remedy, and the four-phase `b128` subtlety. |
| `technique-swizzling` | The NVIDIA-lane equivalent; the reasoning about coupling layout to the consuming instruction is the same. |
| `hw-lds` | Capacity, banking, and the `lgkmcnt` accounting behind the symptom. |
| `technique-in-register-transpose` | When the conflict comes from a transpose and LDS is the binding resource — the RDNA4 situation. |

## Caveats

**A swizzle is not unconditionally a win.** In `kernel-cdna4-fp8-gemm`'s measured
ladder the LDS-swizzle rung *regressed* slightly, 506.70 -> 497.43 TFLOP/s, and
only became valuable after double buffering (1166.41) raised the LDS pressure the
conflicts were throttling. Measure the swizzle in the pipeline it belongs to, not
in isolation.

**Check the ordering first.** If loads are not yet `global_load_dwordx4`, bank
conflicts are not the binding constraint: vectorization was worth 11.2x in that
same ladder against the swizzle's 0.98x. `pattern-memory-bound` comes before this
page.

**Verify against a counter, not a theory.** A swizzle wrong by one shift produces
identical output and none of the speedup. Confirm with `rocprofv3` / ROCm Compute
Profiler that the conflict rate actually fell.
