---
id: lang-amdgcn-asm
title: "AMDGCN assembly and inline asm"
type: language
tags: [amdgcn-asm, sched-barrier, lds, mfma, wmma]
related: [lang-hip, hw-amd-memory-ops, hw-lds, technique-instruction-scheduling-amd, technique-occupancy-tuning-amd]
sources: [doc-rocm-workload-optimization, doc-llvm-amdgpu-usage, blog-rocm-memory-scheduling, blog-rocm-fp8-gemm-cdna4]
reproducibility: snippet
architectures: [gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4]
confidence: source-reported
aliases: [AMDGCN, amdgcn, "GCN assembly", "ISA dump"]
---

# AMDGCN assembly and inline asm

On AMD the ISA listing is not an advanced topic — it is the primary debugging
surface. `doc-rocm-workload-optimization`'s optimization procedure is built
around reading it, because the facts you need (did the loads vectorize, did the
loop pipeline, how many registers did this cost) are not observable any other
way.

## Reading the dump

```bash
# Emit the ISA the compiler actually produced.
AMDGCN_ENABLE_DUMP=1 ./my_kernel 2>&1 | tee kernel.isa
hipcc --offload-arch=gfx1201 --save-temps -c kernel.hip   # or keep the .s

# The four things to check, per the ROCm workload-optimization checklist.
grep -E 'global_load_dword(x4)?'  kernel.isa   # want x4, not scalar
grep -E 'ds_(read|write)_b(32|64|128)' kernel.isa  # want b128
grep -E 's_waitcnt' kernel.isa                 # non-zero operands == pipelined
grep -E '\.vgpr_count|\.agpr_count|\.sgpr_count' kernel.isa
```

The instruction vocabulary that matters:

| Class | Instructions | Counted by |
|---|---|---|
| Device memory | `global_load_dwordx4`, `buffer_load_dwordx4`, `global_store_*` | `vmcnt` |
| LDS | `ds_read_b128`, `ds_write_b128`, `ds_read_tr16_b64` (CDNA4) | `lgkmcnt` |
| Matrix | `v_mfma_*` (CDNA), `v_wmma_*` / `v_swmmac_*` (RDNA) | — |
| Accumulator moves | `v_accvgpr_read`, `v_accvgpr_write` (CDNA) | — |
| Sync | `s_waitcnt`, `s_barrier`, `s_setprio` | — |

A `s_waitcnt vmcnt(0)` immediately before every use of a loaded value means the
loop did **not** pipeline: the wave is issuing one load and blocking on it. A
well-pipelined loop shows non-zero counts, because N loads are outstanding and
the wave is consuming the oldest (`hw-amd-memory-ops`).

`v_accvgpr_read` / `v_accvgpr_write` appearing inside a CDNA inner loop means the
accumulator is being shuffled between register classes — see
`hw-mfma-cdna` on keeping the C tile resident in AGPRs.

## Inline asm

Prefer the `__builtin_amdgcn_*` builtins; they are typed, they participate in
register allocation, and the compiler keeps them correct across targets. Inline
asm is for the cases with no builtin — usually a specific wait count or a fence
the compiler will not emit where you want it.

```cpp
// A hand-placed wait count. Note the "" clobber list: without it the compiler
// may move memory operations across the asm block and the wait guards nothing.
__device__ __forceinline__ void wait_vmcnt_1() {
  asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
}

// Read the hardware wave id --- an SGPR-class read with no builtin equivalent.
__device__ __forceinline__ unsigned hw_reg_wave_id() {
  unsigned v;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(v));
  return v;
}
```

Two rules that are easy to get wrong: `volatile` plus a `"memory"` clobber on
anything ordering-sensitive, and `"s"` versus `"v"` constraints — an SGPR
operand given a `v` constraint forces a needless broadcast, and a divergent
value given `s` is a miscompile.

## Selection guidance

Read the ISA on every kernel you care about; write inline asm rarely. If you
find yourself hand-writing an instruction sequence, check first whether
`technique-instruction-scheduling-amd`'s scheduling builtins express the same
intent — `blog-rocm-fp8-gemm-cdna4` reaches within 3% of hipBLASLt using
`sched_barrier` and `s_setprio` from HIP, with no inline asm at all.
