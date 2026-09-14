---
id: blog-rocm-memory-scheduling
title: "Memory Instruction Scheduling for Lock-Stepped Kernels on AMD Instinct MI300X"
author: "Akash Dutta, Hideki Saito Ido, Michael Selehov, David Tanner (ROCm Blogs)"
url: https://rocm.blogs.amd.com/software-tools-optimization/scheduling_memory_ops_gfx942/README.html
source_category: benchmark-blog
architectures: [gfx942, cdna3]
tags: [instruction-scheduling, lds, agpr, mfma, pipeline-stages]
retrieved_at: 2026-09-14
---

# Memory Instruction Scheduling for Lock-Stepped Kernels on MI300X

Series opener, published 2026-08-17. Establishes the data path and the wait-count
discipline that every hand-scheduled CDNA GEMM inherits.

## Source-backed topics

- **The running example** is a three-stage software-pipelined GEMM. Per K-tile
  each lane loads 256 B from device memory with 16 `buffer_load_dwordx4`
  instructions into VGPRs, stages through a 64 KB LDS region with
  `ds_write_b128` / `ds_read_b128` pairs, and feeds MFMA. Accumulators stay in
  AGPRs (`.agpr_count: 256`) with zero spills, so the C array never generates
  load/store traffic inside the loop. The end-to-end path is HBM -> VGPR -> LDS
  -> AGPR.
- **Wait-count roles**: `vmcnt` is the vector-memory counter for outstanding
  global/buffer loads; `lgkmcnt` is "the LDS counter tracking outstanding LDS
  traffic". Every MFMA waits on `lgkmcnt` so it cannot consume an operand that
  `ds_read_b128` has not returned; the `ds_write_b128`s are drained with
  `s_waitcnt lgkmcnt(0)`.
- **Two `s_barrier`s per iteration**: the first protects the shared LDS region
  before it is overwritten (every wave has finished reading the previous
  K-tile); the second protects it after writing (every wave has finished writing
  before any wave reads).
- **Why lock-stepping is the problem**: waves in matrix-multiply kernels issue
  memory operations in sync, contend for shared resources, and leave bandwidth
  unused.
- **Methodology**: Advanced Thread Trace (ATT) via ROCProfiler and the
  ROCProfiler Compute Viewer, giving a cycle-by-cycle view of one wavefront.
  Trace classes: VMEM (`global_load_dwordx4`, `buffer_load_dwordx4`), LDS, and
  ALU split into SALU / VALU / MFMA. Line colours separate issue stalls,
  programmed `s_waitcnt` waits, and actual issue.
- **Stall categories the series analyses**: device-memory load issue stalls
  (Part 1), LDS write stalls on `ds_write_b128` (Part 2), then LDS reads and
  device-memory writes.

## Boundary

This opener names no scheduling intrinsic. It does not mention
`sched_barrier`, `sched_group_barrier`, or `iglp_opt`, and it publishes no mask
values.
