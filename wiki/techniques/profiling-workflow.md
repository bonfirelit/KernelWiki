---
id: technique-profiling-workflow
title: "Profiling Workflow on ROCm: counter to diagnosis"
type: technique
architectures: [gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4]
tags: [profiling, occupancy-tuning, lds, mfma, instruction-scheduling]
confidence: source-reported
reproducibility: snippet
prerequisites: [pattern-memory-bound]
related: [pattern-lds-bank-conflicts, pattern-register-pressure, technique-occupancy-tuning-amd, pattern-pipeline-stalls]
sources: [blog-rocm-kernel-wiki, blog-amdgpu-kernel-opt-guide, blog-rocm-occupancy-mi355x, doc-rocm-workload-optimization]
symptoms: [low-compute-utilization, memory-bound, register-pressure, bank-conflicts]
---

# Profiling Workflow on ROCm: counter to diagnosis

The loop is top-down and it is a loop: find the hot kernel, collect raw
counters on it, derive the diagnosis, fix, re-measure from the top. The
expensive mistakes are skipping step 1 (tuning a kernel that does not matter)
and stopping after step 2 (a counter is a symptom, not a diagnosis).

## The loop

```bash
# 1. Which kernels dominate? Cheapest pass, no counters, least perturbation.
rocprofv3 --kernel-trace --stats -f csv -d out -- python my_kernel.py

# 2. Raw counters on the hot dispatch. Replay-based: every counter group
#    re-runs the kernel, so keep the list short and collect in a separate run.
rocprofv3 --pmc LDSBankConflict MemUnitBusy VALUBusy MeanOccupancyPerCU \
    -f csv -d out -- ./my_app

# 3. Derived metrics + roofline for the "why".
rocprof-compute profile --name hot_kernel -- ./my_app
rocprof-compute analyze --path workloads/hot_kernel/<arch>
```

Steps 1-2 are `rocprofv3`; step 3 is `rocprof-compute`
(`doc-rocm-workload-optimization` drives this same loop for Triton kernels).
Then take the diagnosis to the pattern pages — `pattern-memory-bound`,
`pattern-lds-bank-conflicts`, `pattern-register-pressure`,
`pattern-pipeline-stalls` — apply the fix, and go back to step 1.

## Counter to diagnosis

The high-signal mappings for CDNA matrix kernels (`blog-amdgpu-kernel-opt-guide`):

| You observe | Diagnosis | Go to |
|---|---|---|
| MFMA-pipe utilization low while memory sits idle | issue-bound: scheduling, not bandwidth | `technique-mfma-pipelining`, `pattern-pipeline-stalls` |
| `LDSBankConflict` non-trivial on a tiled kernel | LDS layout needs swizzle/padding | `technique-lds-bank-conflict-avoidance` |
| `MemUnitStalled` high, bandwidth near sustained peak | memory- or latency-bound | `pattern-memory-bound` |
| `MeanOccupancyPerCU` low and register-limited | VGPR/AGPR pressure, maybe spills | `pattern-register-pressure`, `technique-occupancy-tuning-amd` |

The first row is the one people get wrong: low MFMA utilization with idle
memory means the matrix cores are starved by issue order, and the fix is
scheduling — not more occupancy (`blog-rocm-occupancy-mi355x` held ~97% of
MXFP8 peak at ~12% occupancy with enough ILP).

## Naming traps

- **Renames.** Omniperf is now `rocprof-compute` and Omnitrace is
  `rocprof-sys` (since ROCm 6.2); old entry points survive as shims on some
  installs, so scripts half-work.
- **Two namespaces that do not intersect.** `--pmc` accepts *raw* counters;
  rocprof-compute reports *derived* metrics (e.g. `MfmaUtil`, the number
  `blog-rocm-occupancy-mi355x` quotes). Paste a derived name into `--pmc` and
  you silently collect nothing.
- **Verify against silicon, not docs.** List what the current GPU actually
  exposes with `rocprofv3-avail list` — on this host (gfx1201, ROCm 7.2.4,
  rocprofv3 1.1.0) the `rocprofv3 --list-metrics` flag from older write-ups
  does not exist. Counter names drift between architectures: gfx1201 exposes
  `LDSBankConflict`, `MeanOccupancyPerCU`, `MemUnitBusy`, `VALUBusy`, but has
  **no** `MFMABusy` (no MFMA unit) and **no** `MemUnitStalled` — stalls show
  up as `WriteUnitStalled` / `WAVE_DEP_WAIT` instead.
- **Check the tools exist before trusting an empty result.** This box has
  `rocprofv3`, `rocprof-compute`, `rocprof-sys-*`, and `rocm-smi` installed —
  but rocprof-compute's Python dependencies are missing here, so its
  collect/analyze paths error out. A silent profiler is worse than none.

## Roofline: which roof, and how high it really is

Judge a kernel against the roof of **its own dtype**: MI300X (gfx942) is
~5.3 TB/s HBM3 and ~1307 TF dense BF16; MI355X (gfx950) is 8 TB/s HBM3E and
~5 PF dense MXFP8 (`blog-rocm-occupancy-mi355x`). A BF16 kernel plotted
against the FP8 roof looks artificially far from the ceiling.

And the ceiling itself is lower than the datasheet: sustained HBM read
measures ~86% of nominal peak — 4.56 of 5.3 TB/s on MI300X-class parts, and
6.2-6.4 of 8 TB/s measured on MI350X-class silicon in
`blog-rocm-kernel-wiki`'s microbench. Practical consequence: treat any "90%
of peak bandwidth" claim with suspicion — it is either quoting the nominal
roof against an unreachable target or measuring wrong. Compare against the
sustained figure before declaring a kernel done.

## Practical notes

- **Warm up.** Profile a steady-state iteration; JIT and allocator warmup
  pollute the first dispatch.
- **Time and count in separate runs.** Counter replay perturbs timing;
  `--kernel-trace` is the timing source of truth.
- **Multi-XCD parts aggregate per-CU counters** across chiplets — a healthy
  average can hide cross-XCD imbalance (`xcd`).

## Transfers to RDNA4?

The tools and the loop transfer unchanged — the verified counter list above is
from this gfx1201 host. What does not transfer is the CDNA keying: no
`MFMABusy`/`MfmaUtil` (RDNA4's matrix path is WMMA, `hw-wmma-rdna4`, with no
direct utilization counter), different stall counters, and far lower roofs —
re-derive the roofline per part rather than carrying CDNA numbers. The two
qualitative lessons do transfer: measure sustained bandwidth instead of
trusting the datasheet roof, and never let a derived-metric name anywhere
near `--pmc`.
