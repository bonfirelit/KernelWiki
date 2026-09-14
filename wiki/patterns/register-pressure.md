---
id: pattern-register-pressure
title: "Register Pressure — Low Occupancy"
type: pattern
tags: [tmem, register-reuse, warp-specialization]
symptoms: [register-pressure, low-occupancy, register-spilling]
candidate_techniques: [hw-tmem, technique-warp-specialization, migration-register-to-tmem, technique-occupancy-tuning-amd, technique-register-budgeting]
related: [pattern-compute-bound, hw-tmem]
sources: [doc-ptx-isa-sm100, pr-vllm-16032, blog-rocm-occupancy-mi355x, doc-rocm-workload-optimization]
---

# Register pressure

High live-register demand can reduce residency or cause local-memory spills. Confirm it with compiler resource output and profiler metrics before changing the kernel; occupancy alone is not a performance objective.

Common contributors include register-resident accumulators, epilogue intermediates, unrolled loop state, and long live ranges across asynchronous stages.

## Candidate responses

| Technique | Scope | Tradeoff |
| --- | --- | --- |
| TMEM accumulators | Fifth-generation Tensor Core path | Moves D out of per-thread registers, but adds explicit allocation, synchronization, and TMEM-to-register epilogue work. |
| Warp specialization | Kernel-specific | Gives roles different register budgets, but adds synchronization and may reduce flexibility. |
| Shorter live ranges or less unrolling | General | Can lower register count at the cost of more instructions or less instruction-level parallelism. |
| Smaller tiles | General | Reduces state per CTA but can reduce reuse or Tensor Core efficiency. |

On `sm_100a`/`sm_100f`, PTX exposes each CTA a TMEM view of 512 columns by 128 lanes of 32-bit cells. Treat the required column count as part of the kernel resource design; do not describe that logical CTA view as a documented physical byte capacity per SM.

## On AMD (CDNA / RDNA4)

Read `.vgpr_count`, `.agpr_count`, and `.sgpr_count` straight out of the ISA
(`AMDGCN_ENABLE_DUMP=1`). **Allocation granularity is per-generation**: MI300X
rounds VGPRs to blocks of 16, so usage of 170 becomes 176 and `176 * 3 > 512`
caps residency at 2 waves/EU; MI355X rounds to groups of 8
(`doc-rocm-workload-optimization`, `blog-rocm-occupancy-mi355x`). Sitting a few
registers above a boundary is the one case where `waves_per_eu` earns its keep.

On CDNA there is a second file: VGPRs and AGPRs share 512 registers per lane with
neither class exceeding 256, and a GEMM that keeps its C tile in AGPRs
(`.agpr_count: 256`, zero spills) removes accumulator traffic from the loop
entirely. gfx1201 has no AGPRs, so accumulators compete with operand staging for
one file — see `migration-cdna-to-rdna4`.

**Before acting on this page, check whether occupancy is on the critical path at
all.** `blog-rocm-occupancy-mi355x` measured an MFMA-bound kernel holding ~97% of
matrix peak down to ~12% occupancy, and concludes the sweet spot for such kernels
"is routinely 2-4 waves/SIMD, not 8". Relieving register pressure by shrinking
tiles can cost more ILP than it buys residency. `technique-occupancy-tuning-amd`
has the full argument.
