---
name: KernelWiki
description: Use when the user asks about optimizing GPU kernels on NVIDIA Blackwell (SM100, B200) / Hopper (SM90, H100) OR on AMD CDNA3/CDNA4 (MI300X, MI355X, gfx942/gfx950) / RDNA3/RDNA4 (gfx1100, gfx1201). NVIDIA side: tcgen05/TMEM/CLC/NVFP4/2-SM cooperative, warp specialization, FlashAttention-4, DeepGEMM, FlashMLA, MoE, grouped GEMM, CuTe-DSL/PTX/Triton, or PR references from CUTLASS/SGLang/vLLM/FlashInfer/PyTorch. AMD side: MFMA/WMMA matrix cores, LDS and bank conflicts, wave32 vs wave64, AGPRs, s_waitcnt vmcnt/lgkmcnt, sched_group_barrier/s_setprio scheduling, occupancy and waves_per_eu, MXFP4/OCP-FP8/E8M0 block scales, HIP/ROCm, Triton-on-ROCm, Composable Kernel, AMDGCN ISA, or CUDA→HIP and CDNA→RDNA4 porting. Do NOT use for generic CUDA/HIP Q&A that is not architecture-specific, host-side framework integration, or distributed systems (DeepEP/EPLB/DualPipe).
argument-hint: "[natural-language-question] | [--tag foo --type kernel] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---

# KernelWiki — GPU Kernel Optimization Wiki (NVIDIA + AMD)

Query a structured, cross-referenced knowledge base of GPU kernel optimization across two vendor lanes: NVIDIA Blackwell (SM100) / Hopper (SM90), and AMD CDNA3/CDNA4 / RDNA3/RDNA4. The repository update date is recorded in `README.md`; run `python3 scripts/repo_status.py` for current corpus counts.

## When To Use This Skill

Trigger this skill when the user asks about:

- **Blackwell/SM100 kernel programming** — tcgen05.mma, TMEM, CLC, 2-SM cooperative, NVFP4, FP8/FP4 block scaling, PDL/GDC
- **Kernel implementations** — FlashAttention-4, DeepGEMM, FlashMLA, NSA, GatedDeltaNet, NVFP4 GEMM/GEMV, fused MoE, gated dual GEMM
- **Performance patterns** — low SM utilization, memory-bound, register pressure, compute-bound, tail effects, pipeline stalls
- **DSLs for Blackwell** — CuTe DSL, CUDA C++ with PTX inline, Triton on Blackwell
- **Hopper → Blackwell migration** — wgmma → tcgen05, register → TMEM accumulators
- **PR references** — "how did vLLM/SGLang/FlashInfer/CUTLASS/PyTorch implement X for SM100?"
- **Competition context** — GPU Mode NVFP4 hackathon and FlashInfer MLSys 2026 task definitions, speed-of-light data, and dated results snapshots; participant write-ups are separate blog sources

AMD lane (RDNA4-first; CDNA carried as the deeper technique reservoir):

- **Matrix cores** — WMMA on gfx12 (`__builtin_amdgcn_wmma_*_w32_gfx12`, 16x16x16, fragment lane mapping), MFMA on CDNA3/4, block-scaled `v_mfma_scale_*`, `ds_read_tr16_b64`
- **Memory** — LDS sizing and bank conflicts, `global_load_dwordx4`, direct-to-LDS (`llvm.amdgcn.raw.buffer.load.lds`), `s_waitcnt vmcnt`/`lgkmcnt`
- **Scheduling and occupancy** — `sched_barrier` / `sched_group_barrier` / `s_setprio`, `waves_per_eu`, the four occupancy limiters, and when high occupancy is the wrong target
- **Narrow precision** — OCP FP8 vs CDNA3's FNUZ, MXFP4/MXFP8, E8M0 block scales
- **Languages** — HIP C++, Triton on ROCm, AMDGCN assembly, Composable Kernel
- **Porting** — CUDA → HIP, and CDNA → RDNA4 (what has no gfx12 counterpart at all)

Do NOT use this skill for:

- Generic CUDA or HIP questions unrelated to a specific architecture's tensor/matrix cores
- Host-side framework integration (model loading, request routing, scheduling policy)
- Distributed systems topics — DeepEP, EPLB, DualPipe are out of scope

## How To Query

All commands below run from the skill directory (the clone root — the directory this `SKILL.md` lives in). The scripts auto-resolve the wiki root; **no environment variable required**.

### Runtime dependencies

The query and maintenance scripts are self-contained. They use the host
PyYAML package when present and fall back to the bundled pure-Python loader and
dumper when it is missing. Invoke them directly with `python3`; do not assume
that `pip`, a virtualenv, network access, or a pre-installed PyYAML package is
available. `requirements.txt` is optional and only installs the faster host
implementation.

### Path 1: Unified search (preferred for natural language)

```bash
python3 scripts/query.py "how to fuse gate-up dual GEMM on Blackwell"
python3 scripts/query.py --tag nvfp4 --type kernel
python3 scripts/query.py --repo cutlass --limit 20
python3 scripts/query.py --symptom tail-effect --compact
python3 scripts/query.py --architecture rdna4
python3 scripts/query.py --architecture MI300X --compact
python3 scripts/query.py --tag mfma --type hardware
python3 scripts/query.py --language hip --compact
```

Filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--limit`, `--compact`, `--paths-only`. `--tag`
and `--architecture` accept aliases — `--tag UMMA` matches `tcgen05`,
`--architecture B200` matches `sm100`, `--architecture MI355X` matches `gfx950`, and family tokens (`blackwell`, `rdna4`, `cdna4`) match every exact target beneath them.

### Path 2: Fetch a specific page by id or path

```bash
python3 scripts/get_page.py kernel-flash-attention-4
python3 scripts/get_page.py pr-cutlass-2472
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
python3 scripts/get_page.py kernel-flash-attention-4 --body-only
```

### Path 3: Regex text search across wiki bodies and PR pages

```bash
python3 scripts/grep_wiki.py "tcgen05" --only wiki
python3 scripts/grep_wiki.py "two-CTA" --only wiki
python3 scripts/grep_wiki.py "nvfp4" "block_scale" --any
```

### Path 4: Pre-built cross-reference indices

Auto-generated under `queries/`:

- `queries/by-architecture.md` — exact SM/gfx targets, family-only lanes (Turing through Blackwell; CDNA2 through RDNA4), and validated-unknown architecture evidence
- `queries/by-problem.md` — symptom → pattern page → candidate techniques
- `queries/by-technique.md` — 22 techniques with architectures, confidence, reproducibility, source count
- `queries/by-hardware-feature.md` — tcgen05/tmem/clc/tma/nvfp4/etc. → related wiki + PR pages
- `queries/by-kernel-type.md` — gemm/attention/moe/mla/gated-delta-net → pages
- `queries/by-language.md` — cute-dsl/cuda-cpp/ptx/triton → guide page + related kernels/sources
- `queries/by-repo.md` — PR pages grouped by source repository

### Path 5: Primer, schema, examples

Companion docs under `references/`:

- `references/primer.md` — topic map: hardware features, techniques, symptoms, canonical page IDs. Read this first when the question is broad.
- `references/schema.md` — condensed frontmatter schema, confidence rules, reproducibility ladder, controlled vocabulary, canonical aliases.
- `references/examples.md` — 10 worked query patterns mapping user questions → command sequences → synthesis.

## Output Pattern

When answering from this KB:

1. **Cite specific pages** with paths (e.g., `wiki/kernels/flash-attention-4.md`) and IDs (`kernel-flash-attention-4`).
2. **Follow `sources:` fields** to trace claims back to PRs/blogs/docs.
3. **Respect confidence levels** — `verified` > `source-reported` > `inferred` > `experimental`. Call out when a claim is `experimental` or `inferred`.
4. **Include code snippets** from wiki pages when they exist — technique/kernel/language pages are guaranteed `snippet`-reproducibility (validator-enforced).
5. **Report performance claims with all six fields** — `gpu`, `dtype`, `shape`, `metric`, `value`, `source_id`.

## Knowledge Base Contents

- Source PR pages, synthesized wiki pages, blog/doc/contest summaries, candidate ledgers, query indices, and artifact bundles.
- **Verbatim upstream asset bundles** in `artifacts/` (PR patches and complete kernel files or excerpts) — pinned to upstream SHAs via `PROVENANCE.yaml`
- **Auto-generated query indices** in `queries/`
- **Controlled vocabulary** (80+ tags) in `data/tags.yaml`, alias map in `data/aliases.yaml`
- **Hybrid version-claim registry** — per-page `version_sensitive: <id>` pointers + `data/version-claims.yaml` central registry, validated for bidirectional consistency
- **Status script** `scripts/repo_status.py` — current corpus counts
- **Validator** `scripts/validate.py` — schema, link, artifact, and ledger checks
- **Two vendor lanes** — the NVIDIA lane is Blackwell-first (Hopper-only wiki pages carry explicit `blackwell_relevance`); the AMD lane is CDNA3/CDNA4 + RDNA3/RDNA4-first (pre-CDNA3-only wiki pages carry `amd_relevance`). Source pages preserve upstream evidence and are exempt in both lanes.
- **AMD PR lane not yet ingested** — the AMD side currently has docs, blogs, and wiki pages but no `sources/prs/` coverage; see `CLAUDE.md` for the ingestion recipe

To refresh the corpus: run `scripts/refresh_candidate_ledger.py`, regenerate PR pages and query indices, then validate.

## Quality Guarantees

- Every `verified` page has official-doc + upstream-code evidence
- Every technique/kernel/language page has a compilable snippet
- Every PR page has `inclusion_reason` and an evidence-backed status; current distribution: 942 merged, 2 closed without merge
- All Hopper-only wiki pages have explicit `blackwell_relevance`, and pre-CDNA3-only AMD wiki pages have explicit `amd_relevance`; source pages are exempt
- Every CDNA wiki page states whether its content transfers to RDNA4
