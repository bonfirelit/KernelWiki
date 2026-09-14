---
id: doc-llvm-amdgpu-usage
title: "LLVM AMDGPUUsage — target processors and address spaces"
url: https://llvm.org/docs/AMDGPUUsage.html
source_category: official-doc
architectures: [gfx90a, gfx942, gfx950, gfx1100, gfx1201, cdna3, cdna4, rdna3, rdna4]
tags: [lds, amdgcn-asm]
retrieved_at: 2026-09-14
---

# LLVM AMDGPUUsage

The authoritative processor table backing this wiki's product-to-`gfx` mapping,
and the address-space table that fixes what "LDS" means to the compiler.

## Stable guidance used by this wiki

- The **Processors** table names each AMD product together with its LLVM AMDGPU
  processor. `scripts/pr_policy.py::AMD_PRODUCT_ARCHITECTURE_MAPPINGS` cites this
  page as its `mapping_source`, and `data/aliases.yaml` mirrors it. No product
  name in this repository is canonicalized by inference.
- **AMDGPU address spaces** used throughout the AMD pages here:

  | # | Name | HSA segment | Hardware | Size (bits) |
  |---|---|---|---|---|
  | 0 | Generic | flat | flat | 64 |
  | 1 | Global | global | global | 64 |
  | 3 | Local | group | **LDS** | 32 |
  | 4 | Constant | constant | same as global | 64 |
  | 5 | Private | private | scratch | 32 |
  | 7 | Buffer Fat Pointer | N/A | N/A | 160 |
  | 8 | Buffer Resource | N/A | V# | 128 |

  Address space 3 (LDS) is a 32-bit space: an LDS pointer is not a global
  pointer, which is why `ds_read`/`ds_write` and `global_load` are distinct
  instruction families rather than addressing modes.
- Streamout register accesses "affect LGKMcnt" — the same counter that
  `s_waitcnt lgkmcnt(n)` drains for LDS traffic.

## Boundary

The scheduling intrinsics (`llvm.amdgcn.sched.barrier`,
`llvm.amdgcn.sched.group.barrier`, `llvm.amdgcn.iglp.opt`) live further down
this same page than the section captured here. This wiki therefore does not
quote their mask table from this source; see
`wiki/techniques/instruction-scheduling-amd.md` for what is and is not
established about them.
