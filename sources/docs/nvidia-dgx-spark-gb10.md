---
id: doc-nvidia-dgx-spark-gb10
title: "NVIDIA DGX Spark (GB10) — product specifications"
url: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
source_category: official-doc
architectures: [sm121]
tags: [cuda-cpp]
retrieved_at: 2026-10-30
---

# NVIDIA DGX Spark (GB10) — product specifications

The DGX Spark product page is the vendor's specification sheet for the GB10
Grace Blackwell superchip: a single package carrying an Arm CPU complex and a
Blackwell GPU over a coherent, unified memory system. It is a product page, not
a programming guide — it gives headline numbers and does **not** state a
compute-capability target or an instruction-level feature list, so any kernel
guidance drawn from it has to be paired with the toolkit on the actual device.

Specifications as published on the page:

| Property | Value |
|---|---|
| Superchip | GB10 Grace Blackwell |
| CPU | 20-core Arm (10× Cortex-X925 + 10× Cortex-A725) |
| GPU | Blackwell architecture, 5th-generation Tensor Cores, 4th-generation RT Cores |
| Tensor performance | up to 1 PFLOP FP4 (the page qualifies this as sparse) |
| Memory | 128 GB LPDDR5x, coherent unified system memory |
| Memory interface | 256-bit |
| Memory bandwidth | 273 GB/s |
| Networking | ConnectX-7 @ 200 Gbps, 10 GbE |
| Power | GB10 TDP 140 W (CPU + GPU); 240 W supply |

Two points the page makes that matter for kernel work:

- The memory is described as **coherent unified system memory** — the CPU and
  GPU share one address space rather than the GPU owning a separate VRAM pool.
  The 128 GB figure is the shared budget, not a dedicated framebuffer.
- The "1 PFLOP" figure is **FP4 with sparsity** and is a theoretical peak. It
  says nothing about which instruction (or target variant) reaches that rate on
  the part; that has to be established from the toolkit.

The page does not name the compute capability. On the hardware covered here the
runtime reports 12.1 (`sm_121`); see `hw-gb10` for the target-level detail.

Page accessed 2026-10-30. Product pages are revised without version stamps, so
treat the numbers as the vendor's current headline claims rather than a frozen
datasheet.
