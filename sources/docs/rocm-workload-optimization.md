---
id: doc-rocm-workload-optimization
title: "AMD Instinct MI300 / MI350 Series workload optimization"
url: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
source_category: official-doc
architectures: [gfx942, gfx950, cdna3, cdna4]
tags: [triton-rocm, occupancy-tuning, mfma, lds]
retrieved_at: 2026-09-14
---

# AMD Instinct MI300 / MI350 Series workload optimization

The ROCm documentation chapter that defines the auto-tunable kernel
configuration surface and the occupancy computation this wiki relies on.

## Stable guidance used by this wiki

- Auto-tunable knobs: `num_stages` (2 for a single GEMM; 1 for two fused GEMMs
  such as Flash Attention; 2 for a GEMM fused with a non-GEMM operator; 1 when
  there is no GEMM), `waves_per_eu`, `BLOCK_M`/`BLOCK_N`/`BLOCK_K`, and
  `matrix_instr_nonkdim` (16 selects `mfma_16x16`, 32 selects `mfma_32x32`).
- VGPR budget and granularity: each EU "has 512 available VGPRs, which are
  allocated in blocks of 16." Worked example: "If the current VGPR usage is
  170, it will be rounded up to 176 due to the allocation granularity" —
  occupancy is then limited to 2 waves per EU "because 176 x 3 > 512". Setting
  `waves_per_eu` to 3 asks the LLVM backend to reduce VGPR usage so three waves
  fit. Only worth doing when occupancy is VGPR-limited and usage sits a few
  registers above a boundary.
- MFMA tile choice: "For GEMM kernels on an MI300X GPU, `mfma_16x16` typically
  outperforms `mfma_32x32`, even for large tile/GEMM sizes."
- Occupancy procedure: read `.vgpr_count` from the ISA, allocated LDS `L` from
  the `triton_gpu.shared =` dump, and waves per workgroup `nW` from
  `triton_gpu.num-warps`; then `occ_lds = floor(65536 / L)` on MI300X (64 KB
  LDS) or `floor(163840 / L)` on MI350X (160 KB), and
  `occ = min(floor(occ_vgpr * 4 / nW), occ_lds) * nW / 4`, where `occ_vgpr * 4`
  is the wave count across all four SIMDs of a CU.
- ISA inspection targets named by the doc: confirm `global_load_dwordx4` is
  used for device loads, that LDS traffic uses the `_b128` forms, and read
  `s_waitcnt` operands — `lgkmcnt(n)` for LDS/GDS/constant/message traffic and
  `vmcnt(n)` for vector memory.
- Dump environment variables: `AMDGCN_ENABLE_DUMP=1` (ISA to stdout),
  `MLIR_ENABLE_DUMP=1`, `TORCH_COMPILE_DEBUG=1`.
- Profiling tools named: `rocprofv3` / ROCm Compute Profiler (`rocprof-compute`),
  ROCm Systems Profiler, and the PyTorch profiler.
