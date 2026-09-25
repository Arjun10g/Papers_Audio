# Sources: Compression Is Part of the Compute

Literature-review notes behind `lectures/compression-is-compute.md`, written by four review agents on 2026-09-25. Confidence tags: [V] verified in the source text, [A] abstract only, [S] secondary source, [I] inference or arithmetic.



---

# GPU / systems fundamentals for "Compression Is Part of the Compute"

Literature notes for the fundamentals half of the lecture. Every number carries a confidence tag:

- **[V]** verified: I read it in the primary source (NVIDIA doc, datasheet, whitepaper, or the paper itself).
- **[S]** secondary: taken from a reputable secondary source, or from a primary source I could only read through a summarizer.
- **[I]** inferred: my own arithmetic on [V]/[S] inputs. The derivation is shown so it can be checked.

Rounding for speech: say "about three hundred", not "295.4".

---

## 0. Source list (with URLs)

| Key | Source | URL |
|---|---|---|
| H100-DS | NVIDIA H100 product page / datasheet table | https://www.nvidia.com/en-us/data-center/h100/ |
| A100-DS | NVIDIA A100 product page / datasheet table | https://www.nvidia.com/en-us/data-center/a100/ |
| HGX | NVIDIA HGX platform page (HGX B200 / B300 tables) | https://www.nvidia.com/en-us/data-center/hgx/ |
| DGXB200 | NVIDIA DGX B200 page | https://www.nvidia.com/en-us/data-center/dgx-b200/ |
| H100-WP | NVIDIA H100 Tensor Core GPU Architecture whitepaper v1.02 (2022; preliminary specs) | https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper (mirror used: https://www.hpctech.co.jp/assets/images/info/catalog/pdf/gtc22-whitepaper-hopper_v1.02.pdf) |
| A100-WP | NVIDIA A100 Tensor Core GPU Architecture whitepaper (2020) | https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf |
| CPG | CUDA C++ Programming Guide (archived v12.6, which still has the classic "Performance Guidelines" and "Compute Capabilities" chapters) | https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html |
| CPG-new | CUDA Programming Guide (restructured, v13.x), sec. 2.3.4 | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html |
| CC | CUDA Programming Guide, 5.1 Compute Capabilities (v13.x) | https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html |
| BPG | CUDA C++ Best Practices Guide | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html |
| NCU | Nsight Compute Kernel Profiling Guide | https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html |
| PTX | PTX ISA (current) | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html |
| DLPERF | NVIDIA Deep Learning Performance: GPU Performance Background | https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html |
| VEC | Luitjens, "CUDA Pro Tip: Increase Performance with Vectorized Memory Access", NVIDIA Developer Blog (2013) | https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/ |
| ROOF | Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", CACM 52(4), Apr 2009. Preprint read. | https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf (DOI 10.1145/1498765.1498785) |
| BRRR | Horace He, "Making Deep Learning Go Brrrr From First Principles" (2022) | https://horace.io/brrr_intro.html |
| KIPPLY | kipply, "Transformer Inference Arithmetic" (Mar 30 2022) | https://kipp.ly/transformer-inference-arithmetic/ |
| SCALE | Austin et al., "How to Scale Your Model", ch. "All About Rooflines" (Google DeepMind, 2025) | https://jax-ml.github.io/scaling-book/roofline/ |
| DBX | Databricks, "LLM Inference Performance Engineering: Best Practices" (2023) | https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices |
| FLASH | Dao et al., "FlashAttention", NeurIPS 2022 | https://arxiv.org/abs/2205.14135 |
| HOROWITZ | M. Horowitz, "1.1 Computing's Energy Problem (and what we can do about it)", ISSCC 2014, pp. 10–14 | https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf |
| HAN | Han, Pool, Tran, Dally, "Learning both Weights and Connections for Efficient Neural Networks", NeurIPS 2015 (Fig. 1 reproduces the Horowitz 45 nm energy table) | https://arxiv.org/abs/1506.02626 |
| FGDRAM | O'Connor et al., "Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems", MICRO-50, 2017 | https://www.cs.utexas.edu/~skeckler/pubs/MICRO_2017_Fine_Grained_DRAM.pdf |
| ELEPH | Kim et al., "Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production" (2022) | https://arxiv.org/abs/2211.10017 |
| MARLIN | Frantar, Castro, Chen, Hoefler, Alistarh, "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs" (2024) | https://arxiv.org/abs/2408.11743 |
| FLUTE | Guo, Brandon, Cholakov, Ragan-Kelley, Xing, Kim, "Fast Matrix Multiplications for Lookup Table-Quantized LLMs" (FLUTE), EMNLP Findings 2024 | https://arxiv.org/abs/2407.10960 |
| MACHETE | L. Wilkinson (Neural Magic), "Introducing Machete, a mixed-input GEMM kernel optimized for NVIDIA Hopper GPUs", Red Hat Developer, Oct 2024 | https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel |

---

## 1. Hardware spec sheet: the numbers to quote

### 1.1 Headline datasheet numbers

| GPU | Dense FP16/BF16 tensor | FP32 (CUDA cores, non-tensor) | HBM bandwidth | Memory | Ridge point (dense FP16 FLOP ÷ bytes) |
|---|---|---|---|---|---|
| A100 40GB SXM | 312 TFLOPS [V: A100-WP] | 19.5 TFLOPS [V] | 1,555 GB/s [V: A100-WP] | 40 GB HBM2 | ≈ 200 FLOP/B [I] |
| A100 80GB SXM | 312 TFLOPS (624 with sparsity) [V: A100-DS] | 19.5 TFLOPS [V] | 2,039 GB/s [V: A100-DS] | 80 GB HBM2e | ≈ 153 FLOP/B [I] |
| H100 SXM | 989 TFLOPS dense. The datasheet prints 1,979 "with sparsity", and dense is half of that. [V: H100-DS] | 67 TFLOPS [V] | 3.35 TB/s [V] | 80 GB HBM3 | ≈ 295 FLOP/B [I]. SCALE also gives ≈ 295. [S] |
| H100 NVL | 835 TFLOPS dense (1,671 sparse) [V] | 60 TFLOPS [V] | 3.9 TB/s [V] | 94 GB | ≈ 214 FLOP/B [I] |
| B200 (per GPU) | ≈ 2.25 PFLOPS dense [I/S: see note] | ≈ 75 TFLOPS [I: 600 TFLOPS ÷ 8 on the HGX B200 table] | 8 TB/s [I: 64 TB/s ÷ 8 on DGX B200, V] | 180 GB [I: 1,440 GB ÷ 8, V] | ≈ 280 FLOP/B (FP16), ≈ 560 (FP8), ≈ 1,125 (FP4) [I] |

**Notes and caveats**

- **Sparsity asterisks.** NVIDIA's H100 datasheet lists tensor-core numbers "with sparsity" (2:4 structured sparsity). The dense figure is half. Many blog posts quote 1,979 TFLOPS for H100 FP16, which is the sparse number. For LLM decode, always use dense. [V]
- **The H100 whitepaper numbers are preliminary.** H100-WP (v1.02) shows preliminary values: 1,000 TFLOPS FP16, 60 TFLOPS FP32, 3,000 GB/s, and "Boost clock: Not finalized". The shipping datasheet says 989 / 67 / 3.35 TB/s. Quote the datasheet. [V]
- **B200.** DGX B200 (8 GPUs) lists 1,440 GB total, 64 TB/s HBM3e, FP8 72 PFLOPS "shown in sparse; dense is ½", and FP4 144 | 72 PFLOPS (sparse | dense) [V: DGXB200]. The HGX B200 table lists FP16/BF16 at 36 PFLOPS for 8 GPUs [V: HGX], so 4.5 PFLOPS per GPU. I infer that this is the sparse figure (it follows the same convention as FP8, and FP16 is half of FP8), giving ≈ 2.25 PFLOPS dense per GPU [I]. Secondary sources agree on 2.25 PF dense FP16 and 4.5 PF dense FP8 [S: Jarvislabs/Spheron blogs]. Some sources say 192 GB. NVIDIA's DGX page implies 180 GB per GPU. Say "about 180–190 GB, 8 TB/s".
- **The ridge-point trend.** The ridge point has moved far to the right: the 2009 Roofline paper's AMD Opteron X2 had a ridge at 1.0 FLOP/byte and the Opteron X4 at 4.4 [V: ROOF]. Today's GPUs sit at ~150–300 in FP16, and 500–1,000+ for FP8/FP4 on Blackwell [I]. Lecture line: "In 2009 a kernel needed one flop per byte to saturate the chip. Today it needs about three hundred."

### 1.2 Per-SM microarchitecture (for the "why dequant must be cheap" argument)

| Item | A100 (sm_80) | H100 SXM (sm_90) | Source |
|---|---|---|---|
| SMs | 108 | 132 | [V: A100-WP, H100-WP] |
| FP32 cores / SM | 64 | 128 | [V: H100-WP Table 3] |
| INT32 cores / SM | 64 | 64 | [V: H100-WP Table 3] |
| Tensor cores / SM | 4 | 4 | [V] |
| Dense FP16 tensor FMA / clock / SM | 1,024 (= 2,048 FLOP) | 2,048 (= 4,096 FLOP) | A100: [V: A100-WP "each SM in A100 delivers 1024 FP16 FMA operations per clock"]. H100: [V: H100-WP says "2x the MMA rates of the A100 SM, clock-for-clock"] → 2,048 [I] |
| L2 cache | 40 MB | 50 MB | [V] |
| Combined L1 + shared memory / SM | 192 KB | 256 KB | [V: H100-WP] |
| Max shared memory / SM | 164 KB | 228 KB | [V: H100-WP Table 3; CC table says 228 KB for 9.0] |
| Register file / SM | 256 KB | 256 KB (33 MB across the chip) | [V] |
| Max registers per thread | 255 | 255 | [V: CC] |
| Shared-memory banks | 32 | 32 | [V: CC] |

Consistency check [I]: 108 SMs × 2,048 FLOP/clk × 1.41 GHz ≈ 312 TFLOPS ✔. 108 × 64 × 2 × 1.41 GHz ≈ 19.5 TFLOPS FP32 ✔. For H100, 132 × 128 × 2 × ~1.98 GHz ≈ 67 TFLOPS FP32 ✔, and 132 × 4,096 × ~1.83 GHz ≈ 989 TFLOPS. The implied tensor clock (~1.83 GHz) is lower than the 1.98 GHz max boost [boost from S]. Don't quote clocks on air. Quote the per-SM ratios instead.

**Aggregate on-chip storage on H100 [I from V]:** registers 132 × 256 KB ≈ 33 MB (the whitepaper lists 33,792 KB); shared memory up to 132 × 228 KB ≈ 30 MB; L2 50 MB. Nice aside: the register file is the biggest fast memory on the chip.

**Shared-memory bandwidth [I]:** 32 banks × 4 bytes per clock = 128 B/clk/SM [V: each bank delivers 32 bits per clock]. A100: 128 × 108 × 1.41 GHz ≈ 19.5 TB/s. This matches FLASH's "SRAM … bandwidth estimated around 19TB/s" [V: FLASH, which itself labels it an estimate]. H100 ≈ 33 TB/s at 1.98 GHz [I]. That makes SRAM roughly 10× HBM bandwidth. FLASH: "The on-chip SRAM is an order of magnitude faster than HBM but many orders of magnitude smaller in size." [V]

---

## 2. The roofline model (Williams, Waterman & Patterson, CACM 2009)

**Verified definitions [V: ROOF]:**

- *Operational intensity* = "operations per byte of DRAM traffic … we measure traffic between the caches and memory rather than between the processor and the caches." They chose "operational" over "arithmetic" intensity because (1) arithmetic intensity usually counts processor↔cache traffic, and (2) the model should work for non-arithmetic operations.
- The bound: **Attainable GFlops/sec = Min(Peak Floating Point Performance, Peak Memory Bandwidth × Operational Intensity)**.
- The plot is log–log: the x-axis is FLOPs per DRAM byte and the y-axis is attainable FLOP/s. Bandwidth is a 45° diagonal, because "bytes per second = (GFlops/second)/(GFlops/byte)". Peak compute is a horizontal line.
- "If we think of operational intensity as a column that hits the roof, either it hits the flat part of the roof, which means performance is compute bound, or it hits the slanted part of the roof, which means performance is ultimately memory bound."
- **Ridge point**: "The x-coordinate of the ridge point is the minimum operational intensity required to achieve maximum performance. If the ridge point is far to the right, then only kernels with very high operational intensity can achieve the maximum performance." The ridge point "suggests the level of difficulty for programmers and compiler writers to achieve peak performance."
- The paper uses sustained (benchmarked) bandwidth, "not the pin bandwidth of the DRAM chips". So a real roofline sits somewhat below the datasheet numbers.
- The paper also defines "ceilings" (lower roofs for missing optimizations such as poor locality or no SIMD). In the lecture, a bad memory layout (uncoalesced or misaligned loads) is literally a lower bandwidth ceiling.

**Spoken version (suggested):**
"Draw two lines. One is flat: the most arithmetic the chip can do per second. The other slopes upward: how much arithmetic you can do if you're limited by how fast bytes arrive. Your kernel lives under whichever line is lower. Where they cross is the ridge point, the number of math operations you must do on every byte you fetch before the math units, not memory, become the limit. For an H100 in FP16, that's about three hundred."

Equivalent NVIDIA phrasing [V: DLPERF]: "arithmetic intensity" versus the processor's "ops:byte ratio". An algorithm is math-limited if its arithmetic intensity exceeds the ops:byte ratio and memory-limited otherwise. DLPERF gives V100 as 125 TFLOPS / ~900 GB/s, which yields an ops:byte of ~139 (and ~40 against its 3.1 TB/s L2).

Other framing sources: BRRR splits the costs into compute, memory bandwidth, and overhead, with a factory (compute) / warehouse (DRAM) / shipping (bandwidth) analogy [S: via summarizer]. SCALE gives H100 ≈ 295 FLOP/B and TPU v5e ≈ 240 [S].

---

## 3. Why LLM decode is memory-bound

### 3.1 The mechanism
- Autoregressive decode produces one token per step per sequence. Each step multiplies the current activation vector (1 × d) by every weight matrix. At batch 1, every weight byte is read from HBM once per token and used in exactly one multiply-add. Weights are too big to stay in the 40–50 MB L2 (an 8B-parameter FP16 model is 16 GB). [I, standard]
- DBX: decode is "bottlenecked by how quickly model parameters load from device memory". [S]
- KIPPLY: at small batch "we may be memory bandwidth bound rather than flops". Using A100's 312 TFLOPS / 1.5 TB/s ≈ 208, "if we're going to compute kv for one token, it'll take the same amount of time to compute for up to 208 tokens!" [S]

### 3.2 Arithmetic intensity of a matrix-vector product [I; matches DLPERF's example]
For y = W·x with W an N × K FP16 matrix:
- FLOPs = 2NK (one multiply and one add per weight).
- Bytes ≈ 2NK (weights) + 2K + 2N (vectors) ≈ 2NK.
- **Intensity ≈ 1 FLOP/byte.** DLPERF lists "Linear layer (4096 outputs, 1024 inputs, batch size 1): 1 FLOP/B, memory-limited" and the same layer at batch 512 as 315 FLOP/B, arithmetic-limited [V: DLPERF table].
- With 8-bit weights the intensity is ≈ 2 FLOP/B, and with 4-bit weights ≈ 4 FLOP/B (the FLOP count is unchanged and the bytes drop).
- Against a ridge of ~300, a GEMV uses roughly **1/300th of H100's tensor math**. The math units are ~99.7% idle [I].

### 3.3 Worked example: tokens-per-second ceiling = bandwidth ÷ bytes read per token [I]
Take Llama-3-8B at ≈ 8.0 B parameters (≈ 8.03 B [S]). Ignore the KV cache and activations, and assume 100% of peak bandwidth.

| Weights | Model bytes | A100 80GB (2.04 TB/s) | H100 SXM (3.35 TB/s) | B200 (8 TB/s) |
|---|---|---|---|---|
| FP16 (16 bit) | ≈ 16 GB | ≈ 127 tok/s | ≈ 209 tok/s | ≈ 500 tok/s |
| INT8 (8 bit) | ≈ 8 GB | ≈ 254 | ≈ 417 | ≈ 1,000 |
| 4-bit + group-128 FP16 scales (≈ 4.125 bit/weight) | ≈ 4.1 GB | ≈ 490 | ≈ 810 | ≈ 1,930 |

- Spoken: "Sixteen gigabytes through a three-terabyte-per-second pipe is about two hundred tokens a second, at best. Shrink the weights four times and the ceiling goes up four times, to about eight hundred."
- **Real systems reach a fraction of this.** DBX defines Model Bandwidth Utilization, MBU = achieved bandwidth ÷ peak, where achieved = (parameter bytes + KV-cache bytes) ÷ time per output token. Its worked example is a 7B FP16 model (14 GB) at 14 ms/token, which is 1 TB/s and 50% MBU on a 2 TB/s part. DBX reports ~60% MBU at batch 1 on H100-80GB and ~55% on 4×A100-40GB [S: via summarizer].
- The KV cache adds bytes per token that grow with context length and are *not* shared across a batch. Keep it as a footnote in this section.

### 3.4 Batching raises intensity: the crossover [I; SCALE and KIPPLY agree]
- With batch B, the same weight bytes serve B rows. FLOPs = 2·B·N·K and weight bytes = b·N·K (b = bytes per weight), so **intensity ≈ 2B / b**. That is ≈ B for FP16, ≈ 2B for INT8, ≈ 4B for INT4 (ignoring activation and KV traffic).
- Compute-bound once 2B/b > ridge, i.e. **B\* ≈ ridge × b / 2**:

| | FP16 weights | INT8 weights | INT4 weights |
|---|---|---|---|
| H100 SXM (ridge ≈ 295) | B\* ≈ 300 | ≈ 150 | ≈ 75 |
| A100 80GB (ridge ≈ 153) | ≈ 150 | ≈ 75 | ≈ 40 |

- SCALE states the same thing: on TPU v5e you become compute-bound once batch > ~240 tokens, and int8 weights with bf16 activations lower that threshold to ~120 [S].
- **Lecture-worthy consequence.** Weight-only quantization speeds decode only while you're memory-bound. Compression *lowers* the batch size at which you hit the compute roof, and past that point the dequantization work competes directly with the matmul. MARLIN's abstract: close to the ideal 4× speedup up to batch 16–32, then "gradually decreasing, but still significant" speedup up to 64–128 [V: MARLIN abstract]. MARLIN's own framing: Ampere GPUs have a FLOP-to-byte ratio of ~100–200 in FP16, so with 4-bit weights you can afford ~25–50 multiply-accumulates per quantized weight and still get ~4× [V: MARLIN §2].
- Caveat to say aloud: attention over the KV cache stays memory-bound regardless of batch, because each sequence has its own cache.

---

## 4. Memory hierarchy and memory transactions

### 4.1 The hierarchy (H100 numbers)
HBM3 (80 GB, 3.35 TB/s) → L2 (50 MB, shared by all SMs) → per-SM L1/shared memory (256 KB combined, up to 228 KB as shared) → registers (256 KB per SM; up to 255 × 32-bit registers per thread) → tensor cores / ALUs. [V: H100-WP, CC] Blackwell adds a dedicated **Tensor Memory** next to the tensor cores (see 6.4).

FLASH's A100 illustration [V]: HBM 40–80 GB at 1.5–2.0 TB/s; on-chip SRAM 192 KB per SM × 108 SMs at "~19 TB/s" (estimated).

### 4.2 Sectors, cache lines, transactions [V]
- NCU: "sector: Aligned 32 byte-chunk of memory in a cache line or device memory." "An L1 or L2 cache line is four sectors, i.e. 128 bytes." "The minimum access size in L1 is one sector" (L2 likewise).
- CPG-new §2.3.4.1: "Global memory is accessed via 32-byte memory transactions. When a CUDA thread requests a word of data from global memory, the relevant warp coalesces the memory requests from all the threads in that warp into the number of memory transactions necessary to satisfy the request."
- CPG (v12.6 §5.3.2): "device memory is accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned." Also: "if a 32-byte memory transaction is generated for each thread's 4-byte access, throughput is divided by 8."

### 4.3 Coalescing [V]
- BPG §10.2.1: "For devices of compute capability 6.0 or higher … the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp."
- The best case (CPG-new) is 32 threads reading consecutive 4-byte words: 128 bytes served by four 32-byte transactions, "100% utilization".
- The worst case: consecutive threads 32 or more bytes apart give "a 32-byte memory transaction for each thread", which is 12.5% efficiency.
- A misaligned but sequential access touches **five** 32-byte segments instead of four [V: BPG §10.2.1.2].
- CPG-new: "Ensuring proper coalescing of global memory accesses is one of the most important performance considerations for writing performant CUDA kernels."
- NCU: the ideal sectors-per-request for 32 threads × aligned 4 B is 4, "as every 8 consecutive threads access the same sector".

### 4.4 Load sizes and natural alignment [V: CPG v12.6 §5.3.2, "Size and Alignment Requirement"]
- "Global memory instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes. Any access … compiles to a single global memory instruction if and only if the size of the data type is 1, 2, 4, 8, or 16 bytes and the data is naturally aligned (i.e., its address is a multiple of that size)."
- "If this size and alignment requirement is not fulfilled, the access compiles to multiple instructions with interleaved access patterns that prevent these instructions from fully coalescing."
- "Reading non-naturally aligned 8-byte or 16-byte words produces incorrect results (off by a few words)."
- cudaMalloc returns memory aligned to at least 256 bytes [V: CPG, BPG].
- **Vectorized loads** [S: VEC]: LDG.E.64 / LDG.E.128 (int2/int4, float2/float4) "reduce instruction count and improve memory bandwidth utilization". With int4 you issue 4× fewer load instructions. Any offset pointer must still be aligned. The trade-off is register pressure.
- cp.async (Ampere) copies only 4, 8, or 16 bytes per thread: "cp-size can only be 4, 8 and 16" [V: PTX].
- Lecture reframe [I]: **a 128-bit (16-byte) load per thread is the natural unit.** One warp-wide 16-byte-per-thread load moves 512 contiguous bytes, which is four full 128-byte lines. A packed format should be designed so each thread's 16 bytes contain exactly what it needs.

### 4.5 Why 3-bit and 6-bit elements are awkward [I from V rules above]
- Hardware loads come only in power-of-two sizes (1/2/4/8/16 B), naturally aligned. There is no 3-bit or 6-bit load.
- **3-bit:** 32 isn't divisible by 3. A 32-bit word holds 10 values plus 2 wasted bits, or you pack perfectly over 96 bits (3 words = 32 values), in which case values straddle word boundaries (e.g. value #10 occupies bits 30–32, split across words 0 and 1). Extracting it takes shifts, masks, and a funnel-shift or OR across two registers.
- **6-bit:** the least common multiple of 6 and 32 is 96 bits, so 16 values fit in 12 bytes. 12 bytes is not a legal single load size, and 12-byte groups don't stay 16-byte aligned. You either pad to 16 bytes (wasting 25%), split the load, or reorganize the bits.
- FLUTE's fix for 3-bit [S: via summarizer; wording checked for plausibility]: "While this could be addressed by padding, this would be inefficient. We instead split the (3-bit) quantized weight into two partitions … one containing the 1-bit portion and the other the 2-bit portion" (bit-slicing into power-of-two planes).
- **Hardware confirmation (Blackwell), a strong lecture point [V: PTX §5.5.1.1]:** the TMA tensor-copy engine has native sub-byte types. `.b6x16_p32`: "sixteen 6-bits of data is copied from global memory to the shared memory with an append of 32-bits of padding". `.b4x16_p64`: sixteen 4-bit values get 64 bits of padding. The reverse type `.b6p2x16` packs 6-bit data "contiguously" when writing back to global memory by discarding 2 bits of padding per element. "The sub-byte types are expected to packed contiguously in the global memory." For these types, the global address and strides must be 32-byte aligned, and there are constraints on box size (e.g. 96 B on sm_100f) [V].
  - `ldmatrix` on sm_100a/sm_120a can take `.b6x16_p32` and `.b4x16_p64` sources and expand them to 8-bit containers: "For 4-bit or 6-bit data, 8-bit element will have 4 bits or 2 bits of padding respectively." [V: PTX ldmatrix]
  - **Takeaway:** even NVIDIA's own FP6/FP4 hardware stores data **dense in HBM** (where bytes cost bandwidth) and **padded and aligned on-chip** (where the tensor core needs regular addresses). The format, the layout, and the hardware path were co-designed. That is the thesis, done in silicon.

---

## 5. Shared memory banks [V]
- CPG-new §2.3.4.2 / CPG CC-8.x: "Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a bandwidth of 32 bits per clock cycle."
- Conflicts [V: CPG v12.6 §5.3.2]: "if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialized … decreasing throughput by a factor equal to the number of separate memory requests. If the number of separate memory requests is n, the initial memory request is said to cause n-way bank conflicts."
- Broadcast [V: CPG CC-5.x+]: "A shared memory request for a warp does not generate a bank conflict between two threads that access any address within the same 32-bit word … for read accesses, the word is broadcast to the requesting threads." BPG adds that "multiple broadcasts from different banks are coalesced into a single multicast."
- Classic fix [V: CPG-new]: pad a 2-D array by one column (`[N][M+1]`) so column accesses hit different banks. Stride 1 or 3 words gives no conflicts, and stride 2 gives a 2-way conflict [V: CPG figure caption].

**Relevance to lookup-table (codebook / non-uniform) dequantization [I, with FLUTE as S]:**
- In a LUT kernel, each thread reads `table[code]` with a data-dependent index, so the addresses in a warp are effectively random.
- A 16-entry FP16 table is 32 bytes = 8 words in 8 different banks. Threads that pick the same word broadcast, and the others hit distinct banks, so it is **conflict-free** [I from the broadcast rule].
- Conflicts appear once the table spans more than 32 words (>128 bytes) and two threads pick different words in the same bank. Examples: a 256-entry FP16 table (512 B = 128 words, 4 per bank), or a "vectorized" pair table (FLUTE makes an alternate table for every pair of 4-bit codes, with 16-bit tuples, i.e. 256 × 4 B = 1 KB = 8 words per bank).
- Expected cost for random indices is multi-way serialization.
- **Mitigation: replicate the table** so different threads read different copies in different banks. FLUTE: "We duplicate the 4-bit vectorized lookup table multiple times, placing copies in different memory banks" [S: FLUTE]. The price is shared-memory capacity, which competes with the tiles that feed the tensor cores.
- Throughput reality check [I]: conflict-free shared-memory throughput is 32 four-byte lookups per clock per SM. On H100 that's ≈ 7.4–8.4 × 10¹² lookups/s chip-wide (at 1.76–1.98 GHz). Streaming 4-bit weights at full HBM speed needs 6.7 × 10¹² weights/s. So **one lookup per weight nearly saturates shared memory**, before any conflicts. That's why FLUTE looks up *pairs* (two weights per 4-byte read).

---

## 6. Tensor cores, fragments, and data paths

### 6.1 mma.sync fragment layouts [V: PTX §9.7.16.5.8]
- `mma.sync.aligned.m16n8k16` (Ampere+) is a warp-wide D = A·B + C with A 16×16, B 16×8, C/D 16×8.
- The operands live in registers, scattered across the 32 lanes in a fixed pattern. "Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds a fragment of the matrix."
  - A (f16/bf16): "four .f16x2 registers, with each register containing two .f16/.bf16 elements" (a0…a7). With groupID = laneid >> 2 and threadID_in_group = laneid % 4: row = groupID for a0,a1,a4,a5, else groupID+8; col = threadID_in_group·2 + (i & 1), plus 8 for i ≥ 4.
  - B: "two .f16x2 registers" (b0…b3). row = threadID_in_group·2 + (i & 1) (+8 for i ≥ 2), col = groupID.
  - C/D (f32): four .f32 registers, same row/col pattern as A's first half.
- **What this means [I]:** a thread's 8 A-values are not contiguous in the matrix. They are two adjacent pairs in row g, and the same in row g+8, in column blocks 0–7 and 8–15. A dequantized weight must end up in *exactly the right register of the right lane*, or you pay for shuffles or a round-trip through shared memory.
- A100-WP: Ampere's tensor core "allows data to be shared across all 32 threads in a warp, compared to 8 threads on Volta". For a 16×16×16 multiply, the m16n8k16 instructions cut register accesses from 80 to 28 and instructions from 16 to 2 versus V100 [V].

### 6.2 ldmatrix [V: PTX]
- "Collectively load one or more matrices from shared memory for mma instruction." Shapes: `.m8n8` (16-bit), with `.x1/.x2/.x4` for 1, 2, or 4 matrices. Optional `.trans` transposes.
- Eight threads each supply the address of one 8×8 matrix row ("Each address corresponds to the start of a matrix row"), and "a group of four consecutive threads loads 16 bytes" per row, naturally aligned.
- The data lands in registers in the mma fragment layout.
- Newer targets (sm_100a/sm_120a, PTX 8.6+) add `.m16n16`/`.m8n16` with 8-bit, 6-bit, and 4-bit sources, expanded into 8-bit containers during the load (see 4.5).

### 6.3 Pre-permuting weights offline so the load *is* the fragment
- MARLIN [V]: "we want to dequantize directly into the right register layout for subsequent Tensor Core calls … we again take advantage of the fact that B can be preprocessed offline and reorganize weights such that the 16-byte vector read by each thread contains precisely its necessary 8 quantized weights of 4 separate 16 × 16 Tensor Core blocks. Additionally, within an INT32, weights are stored interleaved, according to the pattern 64207531." MARLIN also reshuffles 16×64 tiles "so that they are laid out contiguously in memory and are thus loaded optimally", and uses cp.async for global→shared copies.
- FLUTE [S]: "offline weight reordering such that after dequantization, the weights are already laid out exactly in the expected format."
- ELEPH [V]: for INT4 they "change the layout of the weights to reduce the number of logic instructions", permuting each group of 8 as [e0,e1,e2,e3,e4,e5,e6,e7] → [e0,e2,e4,e6,e1,e3,e5,e7].
- **Thesis line:** the on-disk bit order is chosen by the kernel's register layout.

### 6.4 Hopper wgmma, TMA; Blackwell tcgen05 [V: PTX, H100-WP]
- **wgmma.mma_async (sm_90a):** a *warpgroup* is "a set of four contiguous warps". "The input matrix A … can be either in registers or in the shared memory. The input matrix B … must be in the shared memory." The shared-memory operands are passed as 64-bit *matrix descriptors*. The instruction is asynchronous (fence / commit_group / wait_group).
  - Consequence [S: MACHETE]: dequantized weights naturally sit in registers, but B must come from shared memory. Machete therefore computes Yᵀ = Wᵀ·Xᵀ "to ensure the weights are 'A' … allowing them to be sourced directly from registers." The kernel's operand-placement rules dictate the math formulation.
- **cp.async (Ampere):** an async global→shared copy that skips the register file. "Async-copy reduces register file bandwidth, uses memory bandwidth more efficiently, and reduces power consumption … can be done in the background while the SM is performing other computations." [V: A100-WP]
- **TMA (Hopper):** "transfer large blocks of data and multi-dimensional tensors from global memory to shared memory and vice-versa … launched using a copy descriptor which specifies data transfers using tensor dimensions and block coordinates instead of per-element addressing." "A single thread creates a copy descriptor … from then on address generation and data movement are handled in hardware." [V: H100-WP]
- **Blackwell (sm_100a) tcgen05:** dedicated Tensor Memory, "512 columns and 128 rows per CTA, with each cell being 32-bits in size" (= 256 KB [I]). `tcgen05.mma` "has single thread semantics, unlike the collective instructions mma.sync or wgmma.mma_async". Accumulators and optionally A live in Tensor Memory [V: PTX §9.7.18].
- The trend [I]: each generation moves operand staging *away* from general-purpose registers and threads (registers → shared memory via descriptors → dedicated tensor memory, with TMA doing address generation). A compressed format must therefore be decodable somewhere on that path: by TMA/ldmatrix natively (Blackwell FP4/FP6), or by CUDA-core code squeezed between shared memory and the MMA.

---

## 7. CUDA-core vs tensor-core throughput: the dequant budget

### 7.1 Chip-level ratios [V datasheets, I ratios]
| GPU | Tensor dense FP16 | CUDA-core FP32 | Ratio |
|---|---|---|---|
| A100 | 312 TFLOPS | 19.5 TFLOPS | 16× |
| H100 SXM | 989 TFLOPS | 67 TFLOPS | ~15× |
| B200 | ~2,250 TFLOPS [I] | ~75 TFLOPS [I] | ~30× (and ~120× against dense FP4 ≈ 9 PFLOPS) [I] |

BRRR [S]: A100 non-matmul ops run at 19.5 TFLOPS versus 312 for matmul ("15x slower"). "Until you're doing about a hundred operations in your unary operator, you'll be spending more time performing memory accesses than actual compute."
A100-WP [V]: non-tensor FP16 is 78 TFLOPS and INT32 is 19.5 TOPS. H100-WP preliminary: non-tensor FP16 120 TFLOPS and INT32 30 TOPS (preliminary numbers, flag them as such).

### 7.2 Per-SM, per-clock instruction throughput [V: CPG v12.6 Table 4 "Throughput of Native Arithmetic Instructions (results per clock cycle per multiprocessor)"]
| Operation | CC 8.0 (A100) | CC 8.6 | CC 8.9 | CC 9.0 (H100) |
|---|---|---|---|---|
| 16-bit FP add/mul/fma (half2 counts as 2 results) | 256 | 256 | 128 | 256 |
| 32-bit FP add/mul/fma | 64 | 128 | 128 | 128 |
| 32-bit integer add/sub | 64 | 64 | 64 | 64 |
| 32-bit bitwise AND/OR/XOR | 64 | 64 | 64 | 64 |
| 32-bit integer shift | 64 | 64 | 64 | 64 |
| compare/min/max | 64 | 64 | 64 | 64 |
| bit-field extract/insert | multiple instr. | multiple instr. | 64 | 64 |
| 8/16-bit int → 32-bit int conversion | 64 | 64 | 64 | 64 |
| **all other type conversions (e.g. int→float I2F)** | **16** | **16** | **16** | **16** |
| warp shuffle | 32 | 32 | 32 | 32 |

Compare [I]: H100 tensor cores do ~2,048 dense FP16 FMAs per clock per SM, versus 64 integer ops, 128 FP32 FMAs, or **16 type conversions**. That's a 32× gap to integer ALU work and a **128× gap to native int→float conversion**.

### 7.3 The dequant instruction budget at batch 1 [I, approximate]
- Streaming 4-bit weights at H100's full 3.35 TB/s means 6.7 × 10¹² weights/s.
- Chip-wide 32-bit integer throughput is 132 SMs × 64 × (1.76–1.98 GHz) ≈ 1.5–1.7 × 10¹³ lane-ops/s. So there are only **~2–2.5 integer lane-instructions per weight** before the integer pipe, not HBM, becomes the bottleneck.
- The FP16 pipe (256 results/clk/SM with half2) adds ~9–10 half-results per weight.
- Native I2F at 16/clk/SM gives ~0.5 per weight, *less than one conversion per weight*. So naive `(float)int4` dequant cannot keep up with HBM [I].
- (The same logic caps shared-memory LUT lookups at ~1.1–1.25 per weight; see §5.)
- This is why production kernels use bit tricks:
  - ELEPH [V]: "the conversion from int to float … was slower than anticipated. In order to improve this, we replaced the native int to float conversion (I2F) with a series of high throughput ALU and FP16 instructions." The trick: OR the integer into the mantissa of FP16 1024.0 (0x6400), then subtract. "0x6400 | Y", "subtract [1152, 1152]" for int8 and [1032, 1032] for int4, two values per 32-bit register, "iterate 4 times since 1 32-bit register holds 8 int4s".
  - MARLIN [V]: "Doing naive type-casts from INT4 to FP16 is slow". Mask plus OR in "a single lop3 instruction", then one FP16 subtract, "dequantize two INT4s in an INT32 at the same time". That is roughly one LOP3 + one HSUB2/HFMA2 per *two* weights, comfortably inside the budget [I].
- At larger batch, each dequantized weight is reused across B rows inside the MMA, so dequant cost per FLOP falls. But the kernel is then near the compute roof, and any CUDA-core work that can't overlap with tensor-core work shows up directly (see §3.4, MARLIN).
- Caveats to state: budget figures assume perfect overlap and ignore address arithmetic, scale loads, loop overhead, and clock throttling. Treat them as order-of-magnitude.

---

## 8. Energy: "bytes are the currency" (optional)

**Horowitz ISSCC 2014** (45 nm, 0.9 V). Text verified [V]:
- "The energy cost of a DRAM access (1 to 2nJ) is a couple of orders-of-magnitude higher than the cost of an internal cache access or functional operation (10pJ)."
- DRAM I/O "takes over 20pJ/bit". Even with better I/O, "the energy cost of a DRAM access will still be large (10pJ/bit, 0.6nJ/8B)".
- A programmable processor has "high energy overhead, 70pJ/instruction, vs. a few pJ for an operation", and "a cache fetch is 20pJ".
- The highest efficiency needs short integers (8–16 bit) and "roughly a thousand operations … completed for each DRAM fetch".

Figure 1.1.9 table (digits are unreadable in the PDF text layer, so these come from HAN Fig. 1, which cites Horowitz) [S]:

| Operation (45 nm) | Energy | Relative |
|---|---|---|
| 32-bit int ADD | 0.1 pJ | 1× |
| 32-bit float ADD | 0.9 pJ | 9× |
| 32-bit register file | 1 pJ | 10× |
| 32-bit int MULT | 3.1 pJ | 31× |
| 32-bit float MULT | 3.7 pJ | 37× |
| 32-bit SRAM cache | 5 pJ | 50× |
| 32-bit DRAM | 640 pJ | 6,400× |

HAN: "Memory access is 3 orders of magnitude more energy expensive than simple arithmetic." Other commonly cited Fig. 1.1.9 values, from memory (not verified in the PDF text; flag [S-]): 8-bit add 0.03 pJ, 8-bit mult 0.2 pJ, 16-bit FP add 0.4 pJ, 16-bit FP mult 1.1 pJ, 8 KB / 32 KB / 1 MB cache 10 / 20 / 100 pJ (64-bit), DRAM 1.3–2.6 nJ (64-bit).

**HBM era [V: FGDRAM, MICRO 2017]:** "The energy to access a bit in HBM2 is approximately 3.97 pJ/bit", and GDDR5 is ~14 pJ/bit. Also: "A future exascale GPU with 4 TB/s of DRAM bandwidth would dissipate upwards of 120 W of DRAM power."
- [I] At ~4 pJ/bit, fetching one FP16 weight costs ~64 pJ, versus a (45 nm) FP16 multiply-add of ~1.5 pJ. Across nodes this is apples to oranges, but it's still a ~40× gap.
- [I] Streaming at H100's 3.35 TB/s at ~4 pJ/bit is ≈ 100 W for HBM access (HBM3 per-bit energy may differ; order of magnitude only).
- Lecture line: "Every weight you *don't* move is energy you don't spend. Compression pays twice, in time and in joules. But only if the decoder is cheaper than the bytes it saves."

---

## 9. Suggested build-up order for the lecture (fundamentals half)
1. **Bytes vs. FLOPs.** Roofline and ridge point (~300 FLOP/B on H100).
2. **Decode is a GEMV.** ~1 FLOP/B, so tokens/s ≈ bandwidth ÷ model bytes (≈ 200 → ≈ 800 tok/s for 8B, FP16 → 4-bit on H100).
3. **So compress the weights?** Only the *bytes moved* matter, and the GPU moves bytes in fixed-shape units: 32-byte sectors, 128-byte lines, coalesced warps, 1/2/4/8/16-byte aligned loads.
4. **Awkward widths.** 3-bit and 6-bit straddle words. Even Blackwell's native FP6 packs densely in HBM and pads to 16 bytes on-chip.
5. **Where decode happens.** In registers on CUDA cores, 15–30× slower than tensor cores, with native int→float at just 16/clk/SM. Budget: ~2 integer ops per weight at batch 1 on H100.
6. **Where the result must land.** The mma fragment layout (and wgmma's "B must be in shared memory"). The weights get pre-permuted offline so a 16-byte load *is* the fragment (Marlin, Elephants, FLUTE, Machete).
7. **LUT formats.** Shared-memory banks, conflicts, replication.
8. **Thesis.** Format × layout × kernel are one design. A compression ratio is not a speedup until the GPU can consume the compressed form at HBM speed.


---

# Quantization formats & algorithms, through the lens of decode cost

Slice for the lecture "Compression Is Part of the Compute".
Guiding question: *what representation moves the fewest bytes per useful computation while still being cheap to decode?*

Confidence legend:
- **[V]** verified in the paper/doc body text (fetched full HTML / official page and read the relevant passage or table)
- **[A]** abstract / landing-page level only
- **[I]** inferred (my own arithmetic or synthesis from verified facts; flagged as such)
- **[S]** from a search-engine snippet or secondary page, not confirmed in the primary doc

Caveat on method: pages were read via a fetch tool that summarises; numbers marked [V] were returned as quotes from the paper's text/tables. Where two fetches disagreed I say so.

---

## 0. The one-paragraph framing

Weight-only quantization (W4A16, W3A16, W2A16…) wins at small batch because single-token decode is memory-bandwidth-bound: fewer bytes per weight means fewer bytes streamed per token. But the tensor cores (pre-Blackwell) cannot multiply INT4/NF4/codebook indices directly, so every weight must be *decoded back to FP16/BF16 in registers* before the MMA. The decode must be cheaper than the bandwidth it saves. Formats form a spectrum:

| Decode cost per weight | Format family | Example |
|---|---|---|
| ~0 (hardware consumes it) | HW block-scaled FP | FP8, MXFP8/6/4, NVFP4 on Blackwell |
| 1 multiply-add (+ bit unpacking) | uniform INT, group-wise scale/zero | GPTQ/AWQ INT4 g128, llama.cpp k-quants |
| 1 small table lookup (+ scale) | non-uniform scalar LUT | NF4, SqueezeLLM |
| lookup in large codebook, several lookups, or trellis walk + inverse Hadamard | vector / codebook / trellis | QuIP#, AQLM, QTIP |

The *algorithms* (GPTQ, AWQ) that pick good values for a standard format are "free" at decode time: the cost is paid once, offline.

---

## 1. Uniform integer, group-wise quantization

**Mechanics.** A group of g consecutive weights (typically g = 64 or 128 along the input dimension) shares a scale s (usually FP16) and optionally a zero point z. Stored integer q in [0, 2^b − 1]; decode is w ≈ s·(q − z) = s·q + (−s·z) — one multiply-add after unpacking bits. [I — standard definition; consistent with GPTQ "FP16 scale and 2-bit zero point per group" quote below]

**Bits-per-weight arithmetic** [I — my arithmetic]:
- 4-bit, g128, FP16 scale only: 4 + 16/128 = **4.125 bpw**
- 4-bit, g128, FP16 scale + FP16 zero: 4 + 32/128 = **4.25 bpw**
- 4-bit, g128, FP16 scale + 4-bit zero: 4 + 20/128 ≈ 4.156 bpw
- 4-bit, g64, FP16 scale (+FP16 zero): 4.25 (4.5) bpw
- 3-bit, g128, FP16 scale + 3-bit zero: 3 + 19/128 ≈ 3.148 bpw

**GPTQ's own accounting** [V]: Frantar et al. write that "group-size 1024 (≈ 0.02 extra bits) improves perplexities by about 0.2 on average and group-size 128 (≈ 0.15 extra bits) by another 0.1", and "At ≈2.2 bit (group-size 128; using FP16 scale and 2-bit zero point per group) the perplexity increase is already less than 1.5 points" (OPT-175B). So a "2-bit" GPTQ model with g128 is really ~2.14–2.2 bpw.
- OPT-175B WikiText2 (Table 5) [V]: GPTQ 4-bit 8.37; 3-bit 8.68; 3-bit/g1024 8.45; 3-bit/g128 8.45 (other datasets improve further at g128: PTB 12.48→12.37, C4 10.47→10.36).

**llama.cpp k-quants** (super-block + sub-block scales; scales themselves quantized) [V, from PR #1684 description, github.com/ggml-org/llama.cpp/pull/1684]:
- Q2_K: "type-1" (scale + min), super-blocks of 16 blocks × 16 weights, 4-bit scales/mins → **2.5625 bpw**
- Q3_K: "type-0" (scale only), 16×16, 6-bit scales → **3.4375 bpw**
- Q4_K: type-1, 8 blocks × 32, 6-bit scales and mins → **4.5 bpw**
- Q5_K: → **5.5 bpw**
- Q6_K: type-0, 16×16, 8-bit scales → **6.5625 bpw**
- Mixes (Q4_K_M, Q3_K_M) use higher-bit types for some tensors (e.g., attention.wv, feed_forward.w2), so file-level bpw is higher still.
- Lecture point [I]: the name "Q2_K" hides that metadata is ~28% of the bits (0.5625/2).

**Kernels: the decode is cheap only if the kernel is well engineered.**
- GPTQ (Frantar, Ashkboos, Hoefler, Alistarh; ICLR 2023; arXiv:2210.17323) [V]: custom "quantized-matrix full-precision-vector product kernel which performs a matrix vector product by dynamically dequantizing weights when needed"; generation is memory-bound. OPT-175B 3-bit, batch 1, avg per-token latency: A100 71 ms, **3.24×** vs FP16 (5 GPUs → 1); A6000 130 ms, **4.53×** (8 → 2). Abstract rounds to 3.25× / 4.5× [A].
- MARLIN (Frantar, Castro, Chen, Hoefler, Alistarh; 2024; arXiv:2408.11743) [A]: FP16×INT4 mixed-precision kernels reach "close to maximum (4×) quantization speedup" for batch sizes 16–32, with gradually smaller speedups at 64–128; up to 2.8× end-to-end in vLLM. Lecture point: as batch grows the matmul becomes compute-bound and the dequant work stops being hidden — the win shrinks. [I for the interpretation; the paper frames it as keeping kernels memory-bound under batching]

---

## 2. Non-uniform / lookup formats

### NF4 (QLoRA)
Dettmers, Pagnoni, Holtzman, Zettlemoyer. "QLoRA: Efficient Finetuning of Quantized LLMs." 2023. arXiv:2305.14314.
- NF4 = 4-bit NormalFloat: code values are quantiles of N(0,1) normalised to [−1, 1]; "information-theoretically optimal for normally distributed weights"; blocksize 64 with an absmax constant per block. [V]
- **Double quantization** [V]: first-level constants are FP32 per 64 weights → 32/64 = **0.5 bits/param**. Quantize those constants to FP8 with second-level blocksize 256 → 8/64 + 32/(64·256) = **0.127 bits/param**. Saves ≈0.37 bits/param, ≈3 GB for a 65B model.
  - So NF4 without DQ ≈ 4.5 bpw; with DQ ≈ 4.127 bpw. [I arithmetic, V components]
- Compute: weights are dequantized from NF4 storage to **BF16** for the matmul. [V] NF4 is a storage format, not a compute format.
- Decode needs a 16-entry lookup (index → NF4 value), then × absmax, where the absmax itself must first be decoded from FP8 × second-level FP32 constant under DQ. [I — follows from the definition; a lookup is required because the 16 levels are not evenly spaced]

### SqueezeLLM
Kim, Hooper, Gholami, Dong, Li, Shen, Mahoney, Keutzer. "SqueezeLLM: Dense-and-Sparse Quantization." ICML 2024. arXiv:2306.07629.
- Motivating claim [A]: memory bandwidth, not compute, is the bottleneck for single-batch generative inference.
- Non-uniform quantization via **sensitivity-weighted k-means** (Fisher-information weights) → per-output-channel LUT with 2^b centroids (8 FP16 values for 3-bit, 16 for 4-bit). [V]
- **Dense-and-sparse**: pull out 0.45% of weights into a sparse FP16 matrix (CSR) — 0.05% sensitive values + 0.40% outliers. [V]
- **Bit accounting** (Table 1) [V]: 3-bit dense-only = **3.02** bits; 3-bit + 0.45% sparse = **3.24** bits; 4-bit + sparse = **4.27** bits. The sparse part costs ~0.2 bpw — outliers are expensive because each needs a value plus an index. [I for the explanation]
- Kernels: "LUT-based kernels", "compressed matrices store 3/4-bit indices, which correspond to LUT entries containing FP16 values"; arithmetic is FP16. [V]
- Speed, A6000, 128 generated tokens [V, Table 3]: LLaMA-7B 3-bit 1.5 s vs 3.2 s FP16 (~2.1×); LLaMA-13B 2.4 s vs 5.6 s (~2.3×). The sparse part adds ~10% latency (still ~2.2×). Lecture point: the sparse outlier path is a separate, irregular kernel — a real decode tax.

### FLUTE (why LUT decode is hard on GPUs)
Guo, Brandon, Cholakov, Ragan-Kelley, Xing, Kim. "Fast Matrix Multiplications for Lookup Table-Quantized LLMs." 2024. arXiv:2407.10960. [V, summarised]
- Problems it names: odd bit-widths (3-bit) need offline bit-slicing to match Tensor Core layouts; dynamic indexing into a shared-memory table is slow; **shared-memory bank conflicts** when threads hit the same bank. Fixes: "vectorized" table (look up pairs of values at once) and duplicating the table across banks.
- Reports 2–4× faster than existing GEMM kernels at batch < 32, group size 128; beats bitsandbytes and BitBLAS NF4; 1.5–2× end-to-end throughput in vLLM.
- Lecture point [I]: a lookup is "one instruction" on paper but its cost is a shared-memory problem; engineering can hide it, but it is not free.

---

## 3. Algorithms that choose values (standard format → free decode)

### GPTQ
Frantar, Ashkboos, Hoefler, Alistarh. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023. arXiv:2210.17323.
- Uses approximate second-order (Hessian) information; quantizes column by column and updates remaining weights to compensate. 175B in ~4 GPU hours; 3–4 bits; also reasonable at 2-bit and ternary (ternary: 9.20 WikiText2 PPL on OPT-175B). [A/V]
- Output is plain INT with group scales — **the decoder does not know GPTQ was used.** [I]

### AWQ
Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024 (**Best Paper Award** — [A], stated on arXiv page). arXiv:2306.00978.
- "protecting only 1% salient weights can greatly reduce quantization error"; salient channels identified from **activation** magnitude, not weights. [A]
- Key hardware argument [V]: "keeping 0.1% of weights in FP16 can improve the quantized performance without a noticeable increase in model size… such a mixed-precision data type will make the system implementation difficult." So AWQ instead **scales up salient input channels** (and folds the inverse into the preceding op) so a uniform INT4 format suffices. Compare SqueezeLLM, which pays for the sparse FP16 side-matrix.
- Default config: INT4, group size 128 (W4A16). [V]
- TinyChat: on-the-fly dequantization fused into the matmul, SIMD-aware weight packing (up to 1.2× on ARM), kernel fusion. 2.7–3.9× over HF FP16 on RTX 4090 (Llama-2, MPT, Falcon); ~3.5× on Jetson Orin. [V]

---

## 4. Vector / codebook / trellis quantization + incoherence processing

### QuIP (background)
Chee, Cai, Kuleshov, De Sa. "QuIP: 2-Bit Quantization of LLMs With Guarantees." NeurIPS 2023. arXiv:2307.13304. Introduced incoherence processing (random orthogonal transforms so no weight is an outlier). [not fetched; citation from knowledge — treat as [I]]

### QuIP#
Tseng, Chee, Sun, Kuleshov, De Sa. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." ICML 2024. arXiv:2402.04396.
- Three ingredients [A]: randomized Hadamard transform (RHT) for incoherence; vector quantization with E8-lattice codebooks ("optimal 8-dimension unit ball packing"); fine-tuning.
- **E8P codebook** [V]: 2^16 entries (8-dim vectors, 16 bits per 8 weights = 2 bpw) but "only requires lookups into a 2^8-entry table" — 8 bits index into a table S, 7 bits of sign flips, 1 bit for ±1/4 shift. Needs "only 1KiB of space and therefore fits in the L1 cache of any modern GPU, even after duplicating for bank conflicts (32×)." **This is a codebook deliberately designed to be decoded from symmetry rather than stored.**
- 3-bit = 2-bit E8P + 1-bit E8 codebook; 4-bit = residual VQ, E8P applied twice. [V] (i.e., 1–2 lookups per 8 weights)
- **Hadamard cost** [V]: RHT is O(n log n) (vs QuIP's Kronecker Θ(n√n)); Hadamard entries are ±1 so no floating multiplies; sign vectors add "less than 0.01 bits per weight". At inference the transform must be applied to activations/outputs online. [I that it is online; standard for weight-only incoherence methods]
- **Speed, RTX 4090** [V, Table 5]: 2-bit Llama-2-70B **32.74 tok/s at 56.84% of peak memory bandwidth**; 2-bit 7B 170.50 tok/s at 29.60%. Table 6: 2-bit 7B QuIP# **106.3 tok/s vs AQLM 20.6 vs FP16 33.1** (different setup from Table 5).
  - Lecture point [I]: even a codebook designed for GPUs only reaches ~57% of bandwidth on 70B and ~30% on 7B — decode overhead and small-kernel effects cap how much of the byte savings becomes speed.
- **Quality** [V]: Llama-2-70B WikiText2 (ctx 2048): FP16 3.32, QuIP# 2-bit 4.16, OmniQuant 2-bit 7.81. A second table reports QuIP# 2-bit = 6.19 at ctx 4096, the same figure QTIP cites. The two tables use different evaluation setups, so don't compare numbers across them. (6.19 > 4.16 is odd, since longer context usually *lowers* perplexity. Probably a different model version or eval protocol. Unresolved; cite each number with its table.)

### AQLM
Egiazarian, Panferov, Kuznedelev, Frantar, Babenko, Alistarh. "Extreme Compression of Large Language Models via Additive Quantization." ICML 2024. arXiv:2401.06118.
- Additive multi-codebook quantization: each group of g=8 weights is the **sum of M codewords**, one from each of M codebooks of size 2^B; codebooks learned per layer, jointly tuned across a block. Claims first Pareto-optimal scheme below 3 bits. [A/V]
- Configs [V]: 2-bit → 1 codebook of 2^15 or 2^16, g=8 ("1×16"); 3-bit → 2 codebooks of 2^12; 4-bit → 2 codebooks of 2^15/2^16. Also faster "2×8" (two 256-entry codebooks).
- Storage [V]: each group of g weights uses M·B bits for codes plus the codebooks cost g·2^B·16 bits (FP16) per codebook.
- Reported avg bits (Table 1) [V]: Llama-2 7B 2.02 (Wiki2 6.59); 13B 1.97 (5.60); 70B **2.07 (3.94)**; vs QuIP# (older version) 7B 2.02 (8.22), 70B 2.01 (4.16). Paper note [V]: "the average bits per parameter takes into account only quantized weights, we do not include parameters kept in floating precision" (embeddings, head, norms).
- **Codebook size vs cache** [I, arithmetic from verified config]: a 1×16 codebook of 8-dim FP16 vectors = 2^16 × 8 × 2 B = **1 MiB per layer** — far bigger than L1/shared memory (~100s of KB per SM), so each group's decode is a random global/L2 gather. 2×8 = 2 × 256 × 8 × 2 B = **8 KiB** — fits in shared memory, but needs 2 lookups + add per 8 weights.
- **Speed** [V, Table 5, RTX 3090, vs FP16]: 1×16: 7B 1.31×, 13B 1.20×, 70B 1.20×. 2×8: 7B 1.57×, 13B 1.82×, 70B **3.05×**. Paper: "Multiple smaller codebooks allow efficient GPU cache utilization, leading to greater speedup, at the price of slightly lower accuracy." CPU (Intel i9, Kx8 codebooks, vs FP32): ~2.75–4× (two fetches returned 3.69× and 4.07× for 70B — likely different configs; treat as "up to ~4×"). GitHub README [V]: CUDA 2×8 "Up to ~3.0x", 1×16 "Up to ~1.3x", Triton "Up to ~0.7x" (i.e., slower than FP16), Numba CPU "Up to ~4.0x".
  - **Lecture gold** [I]: the most accurate AQLM variant (1×16) is barely faster than FP16 despite moving ~8× fewer weight bytes — the decode ate the savings. The fast variant trades accuracy for a cache-resident codebook.
- Bookkeeping example [V/I]: README lists Llama-2-7b AQLM 1×16 at **2.4 GB** (Wiki2 PPL 5.92, PV-tuned table) and 2×8 at 2.2 GB. 2.4e9 B × 8 / 6.74e9 params ≈ **2.85 bits per parameter** for a "2-bit" model [I], because FP16 embeddings/LM head (~0.5 GB for 32k×4096×2) + codebooks + scales ride along.

### QTIP
Tseng, Sun, Hou, De Sa. "QTIP: Quantization with Trellises and Incoherence Processing." NeurIPS 2024 (Spotlight). arXiv:2406.11235.
- Idea [A]: replace VQ with **trellis-coded quantization** — "a stateful decoder that separates the codebook size from the bitrate and effective dimension", enabling "ultra-high-dimensional quantization" that VQ can't do (VQ codebooks grow exponentially with dimension, so VQ is stuck ≤ 8 dims).
- **Bitshift trellis** [V]: (L, k, V) = (16, 2, 1 or 2): the decoder state is a 16-bit window sliding over the bitstream by k·V bits per step, so decoding any position is a shift-and-mask — parallel, no sequential Viterbi at inference. [I for the "shift-and-mask/parallel" paraphrase; parameters V]
- **Computed codes** — turn the 16-bit state into a pseudo-Gaussian value with a few ALU ops instead of a table [V]:
  - **1MAD**: a multiply-add + mask (~2 core instructions, ~4 total per weight as summarised)
  - **3INST**: "can be implemented in 3 ALU instructions" per weight
  - **HYB** (hybrid): "an amortized 2 instructions per weight" plus a 2^Q × 2 lookup table; with Q = 9 a **2 KiB codebook, which fits in L1**. A pure-lookup variant would need ~32 KB.
- **Speed** [V, Table 4, RTX 6000 Ada, 960 GB/s], batch-1 decode, 2-bit: Llama-2-7B **QTIP 188 tok/s vs QuIP# 186 vs AQLM 81.5 vs FP16 55.9**; Llama-2-70B **QTIP 23.5 vs QuIP# 22.2 vs AQLM 8.78**. Same speed as QuIP# with higher quality.
- **Quality** [V, Table 5]: 2-bit Llama-2-70B Wiki2 / C4 = QTIP **5.86 / 7.73**, QuIP# 6.19 / 8.16, AQLM 6.14 / 8.09 (ctx 4096).
- Lecture point [I]: QTIP is the clearest example of *designing the code for the decoder*: the "codebook" is computed from bits with 2–3 integer instructions, so a far better (high-dimensional) quantizer costs about the same to decode.

---

## 5. Hardware low-precision floating point (decode cost → ~0)

### FP8
Micikevicius et al. (15 authors, NVIDIA/Arm/Intel). "FP8 Formats for Deep Learning." 2022. arXiv:2209.05433. [A]
- **E4M3**: 4 exp, 3 mantissa; extends range by having no infinities and a single NaN mantissa pattern (max normal 448 — [I from knowledge; not quoted in fetch]).
- **E5M2**: 5 exp, 2 mantissa; follows IEEE-754 special-value conventions.
- Matched 16-bit training quality across CNNs/RNNs/Transformers up to 175B; also PTQ of models that resisted INT8. [A]
- Native on Hopper (H100) and later tensor cores. [I, common knowledge]

### OCP Microscaling (MX) formats
Rouhani et al. (Microsoft, AMD, Intel, Meta, NVIDIA, Qualcomm). "Microscaling Data Formats for Deep Learning." 2023. arXiv:2310.10537. Plus OCP "Microscaling Formats (MX) Specification v1.0" (opencompute.org — PDF returned 403 to my fetch; E8M0 details below from search snippet [S]).
- Block = one shared scale X + k elements P_i; value = X·P_i; **k = 32**. [V]
- Concrete formats (Table 1) [V]: all use **E8M0** 8-bit scale, block 32:
  - MXFP8: FP8 E4M3 or E5M2 (8 bits)
  - MXFP6: FP6 E2M3 or E3M2 (6 bits)
  - MXFP4: FP4 E2M1 (4 bits)
  - MXINT8: INT8
- E8M0 [S]: unsigned biased exponent (bias 127), pure power of two, no Inf, one NaN encoding (0xFF); a NaN scale makes the whole block NaN.
- **Effective bits** [I]: element + 8/32 → MXFP4 **4.25**, MXFP6 6.25, MXFP8/MXINT8 8.25 bpw. Confirmed for MXFP4 by the gpt-oss model card (below). [V]
- Power-of-two scale means applying it is an exponent add — trivially cheap in hardware. [I]

### NVFP4
NVIDIA Technical Blog, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference" (2025). developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ [V]
- Elements FP4 **E2M1** (values ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}); **16-element micro-block** with one **FP8 E4M3** scale; plus a second-level **FP32 per-tensor** scale.
- vs MXFP4: half the block size (16 vs 32) and a non-power-of-two scale (E4M3 vs E8M0) → finer, more accurate scaling.
- **4.5 bits per value** including scales (blog states this). [V] (4 + 8/16; per-tensor FP32 negligible [I])
- Memory: ~3.5× smaller than FP16, ~1.8× smaller than FP8. [V]
- Accuracy: DeepSeek-R1-0528 FP8 → NVFP4 "1% or less" degradation across benchmarks (AIME 2024 +2%). [V — vendor claim]
- Implemented by Blackwell fifth-gen Tensor Cores. [V]

### What Blackwell tensor cores natively consume
CUTLASS docs, "Blackwell SM100 GEMMs" (docs.nvidia.com/cutlass/.../blackwell_functionality.html) [V]:

| tcgen05.mma kind | A/B | Throughput (as documented, relative to Hopper) | Scale type | Scale block |
|---|---|---|---|---|
| kind::tf32 | tf32 | 2× Hopper | — | — |
| kind::f16 | f16/bf16 | 2× Hopper | — | — |
| kind::i8 | int8 | 2× Hopper | — | — |
| kind::f8f6f4 | mixed f4/f6/f8 (unscaled) | 2× Hopper | — | — |
| kind::mxf8f6f4 | MXFP8/6/4 | 2× Hopper | ue8m0 | 32 |
| kind::mxf4 | MXFP4 × MXFP4 | **4× Hopper** | ue8m0 | 32 |
| kind::mxf4nvf4 | MXFP4 / NVFP4 | **4× Hopper** | ue4m3 (or ue8m0) | 16 / 32 |

- The block scale is applied *inside* the MMA instruction: "the operand data is multiplied by a scale factor before multiply-add" (Colfax CUTLASS block-scaling tutorial) — Hopper needed CUDA cores for this. [V]
- Subtlety [I from the table]: FP4 only gets the doubled rate through the FP4-only block-scaled kinds; FP4 in the mixed f8f6f4 kind runs at FP8 rate. FP6 runs at FP8 rate.
- PTX [S]: mxf4nvf4 supports block16 (scale_vec::4X) with E4M3 or E8M0 and block32 (2X) with E8M0 only.
- SM120 (GeForce RTX 50 / RTX PRO Blackwell) supports the same narrow types, with restrictions (TN layout only, cluster 1×1×1). [V, CUTLASS docs]

**Datasheet throughput** (NVIDIA HGX page, 8-GPU systems) [V]:
- HGX B200: FP4 144 PFLOPS sparse | **72 dense**; FP8/FP6 72 sparse (**36 dense**); INT8 72 POPS sparse; FP16/BF16 36 sparse (**18 dense**); TF32 18 sparse.
  - Per-GPU [I, ÷8]: FP4 9 dense, FP8 4.5 dense, FP16 2.25 dense PFLOPS → **FP4 : FP8 : FP16 = 4 : 2 : 1**.
- HGX B300 (Blackwell Ultra): FP4 144 sparse | **108 dense**; FP8/FP6 72 sparse; INT8 **3 POPS**; FP16/BF16 36; TF32 18. → dense FP4 is **3× dense FP8** on B300, and INT8 tensor throughput is cut drastically — the hardware is betting on block-scaled FP4 over INT. [V numbers, I interpretation]
- RTX 50 [S]: RTX Blackwell adds FP4 and FP6 tensor ops to Ada's FP16/BF16/TF32/INT8/INT4/FP8; RTX 5090 marketed at 3,352 "AI TOPS" (NVIDIA blog; presumably FP4 with sparsity — not verified).

**gpt-oss as a real deployment example** (OpenAI, "gpt-oss-120b & gpt-oss-20b Model Card," 2025, arXiv:2508.10925) [V]:
- "We post-trained the models with quantization of the MoE weights to MXFP4 format, where weights are quantized to 4.25 bits per parameter." MoE weights are 90+% of parameters; 120b fits a single 80 GB GPU.
- 116.8B total / 5.1B active params, **60.8 GiB** checkpoint; 20b: 20.9B / 3.6B, 12.8 GiB.
- Bookkeeping [I]: 60.8 GiB × 8 / 116.8e9 ≈ **4.47 bits per parameter overall** — the non-MoE ~10% stays in BF16 and pulls the average up from 4.25.

---

## 6. Activation quantization (contrast only)

Weight-only schemes dequantize to FP16 and run an FP16 matmul — they save bytes, not FLOPs. W8A8/W4A4 run the matmul itself in low precision, which matters once you're compute-bound (large batch, prefill).

- **SmoothQuant** — Xiao, Lin, Seznec, Wu, Demouth, Han. ICML 2023. arXiv:2211.10438. [A] Migrates quantization difficulty from activations (outlier channels) to weights with an offline per-channel scaling, enabling W8A8 INT8 GEMMs; up to **1.56× speedup, 2× memory reduction**; 530B model on a single node. (Authors from knowledge [I].)
- **QuaRot** — Ashkboos, Mohtashami, Croci, Li, Cameron, Jaggi, Alistarh, Hoefler, Hensman. 2024. arXiv:2404.00456. [V] Hadamard rotations of hidden states/activations/KV cache so "all matrix multiplications are performed in 4 bits, without any channels identified for retention in higher precision." Llama-2-70B W4A4KV4: ≤0.47 WikiText-2 PPL loss, 99% zero-shot retained. RTX 3090: up to **3.33× prefill** speedup (bs 64, seq 2048), **3.89×** decode memory savings; 4-bit linear layer 3.2× (7B) / 4.3× (70B) vs FP16; online Hadamard adds "at most 7% overhead".
- **SpinQuant** — Liu et al. (Meta), 2024, arXiv:2405.16406. [A] Learned rotations; W4A4KV4 Llama-2-7B gap to FP only 2.9 points; beats SmoothQuant by 25.0, LLM-QAT by 19.1 points; up to 45.1% smaller gap than QuaRot on Llama-3-8B.
- Tie-in [I]: rotation (Hadamard) shows up in both worlds — QuIP#/QTIP use it to make weights codebook-friendly; QuaRot/SpinQuant use it to make activations INT4-friendly. Either way it's an online transform that costs a little compute to make the representation cheaper.

---

## 7. Bits-per-weight bookkeeping — collected examples

| Nominal | Actual | What the extra is | Source / conf. |
|---|---|---|---|
| INT4 g128, FP16 scale | 4.125 bpw | 16-bit scale / 128 | arithmetic [I] |
| INT4 g128, FP16 scale+zero | 4.25 bpw | +16-bit zero / 128 | [I] |
| GPTQ 3-bit g128 | ≈3.15 bpw | "≈0.15 extra bits" | GPTQ paper [V] |
| GPTQ "2-bit" g128 | ≈2.2 bpw | FP16 scale + 2-bit zero per 128 | GPTQ paper [V] |
| NF4, no double quant | 4.5 bpw | FP32 absmax per 64 = 0.5 bpw | QLoRA [V] |
| NF4 + double quant | ≈4.127 bpw | FP8 absmax/64 + FP32/(64·256) | QLoRA [V] |
| llama.cpp Q2_K | 2.5625 bpw | 4-bit scale+min per 16, super-block FP16 | PR #1684 [V] |
| llama.cpp Q3_K | 3.4375 bpw | 6-bit scales per 16 | [V] |
| llama.cpp Q4_K | 4.5 bpw | 6-bit scale+min per 32 | [V] |
| SqueezeLLM 3-bit | 3.02 → 3.24 bpw | LUTs; 0.45% sparse FP16 outliers (CSR) | SqueezeLLM Table 1 [V] |
| SqueezeLLM 4-bit + sparse | 4.27 bpw | same | [V] |
| MXFP4 | 4.25 bpw | E8M0 per 32 | spec/gpt-oss card [V] |
| NVFP4 | 4.5 bpw | E4M3 per 16 (+FP32/tensor) | NVIDIA blog [V] |
| gpt-oss-120b "MXFP4" | ≈4.47 bits/param whole checkpoint | non-MoE weights in BF16 | model card [V] + arithmetic [I] |
| AQLM "2-bit" Llama-2-7B 1×16 | 2.02 reported; 2.4 GB file ≈ 2.85 bits/param | FP16 embeddings/head excluded from reported bits; 1 MiB codebook per layer | paper [V], README [V], arithmetic [I] |
| QuIP# 2-bit | ~2 bpw + <0.01 for RHT sign vectors | E8P codebook shared (1 KiB) | QuIP# [V] |

---

## Takeaway lines for the lecture (my synthesis [I])

1. Weight-only quantization buys bandwidth, and the price is decode work in the matmul inner loop. The format decides that price, not the algorithm.
2. The best algorithms (GPTQ, AWQ) are "free" because they keep the format boring. AWQ explicitly rejects mixed precision *because* it's hard to implement efficiently.
3. The strongest 2-bit methods won by designing codebooks for the decoder: QuIP# (1 KiB E8P from lattice symmetry), QTIP (codes computed in 2–3 instructions). AQLM's most accurate 1×16 mode moves ~8× fewer weight bytes than FP16 but runs only ~1.2–1.3× faster.
4. Metadata is real: scales, zeros, LUTs, outliers and unquantized layers routinely add 0.1–0.5 bits per weight, and more at the whole-file level ("2-bit" 7B ≈ 2.85 bits/param on disk).
5. Blackwell ends the argument for its own formats: MXFP4/NVFP4 block scales are applied inside the tensor core, dense FP4 is 2× FP8 and 4× FP16 on B200 (3× FP8 on B300), and decode cost goes to ~zero. The representation *is* the compute.

## Full citation list
- Frantar, Ashkboos, Hoefler, Alistarh. GPTQ. ICLR 2023. arXiv:2210.17323
- Frantar, Castro, Chen, Hoefler, Alistarh. MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs. 2024. arXiv:2408.11743
- Lin et al. (Song Han group). AWQ. MLSys 2024 Best Paper. arXiv:2306.00978
- Dettmers, Pagnoni, Holtzman, Zettlemoyer. QLoRA. NeurIPS 2023. arXiv:2305.14314
- Kim, Hooper, Gholami, Dong, Li, Shen, Mahoney, Keutzer. SqueezeLLM. ICML 2024. arXiv:2306.07629
- Guo, Brandon, Cholakov, Ragan-Kelley, Xing, Kim. FLUTE. 2024. arXiv:2407.10960
- Chee, Cai, Kuleshov, De Sa. QuIP. NeurIPS 2023. arXiv:2307.13304 (not fetched)
- Tseng, Chee, Sun, Kuleshov, De Sa. QuIP#. ICML 2024. arXiv:2402.04396
- Egiazarian, Panferov, Kuznedelev, Frantar, Babenko, Alistarh. AQLM. ICML 2024. arXiv:2401.06118; code github.com/Vahe1994/AQLM
- Tseng, Sun, Hou, De Sa. QTIP. NeurIPS 2024. arXiv:2406.11235
- Micikevicius et al. FP8 Formats for Deep Learning. 2022. arXiv:2209.05433
- Rouhani et al. Microscaling Data Formats for Deep Learning. 2023. arXiv:2310.10537; OCP MX Spec v1.0 (opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- NVIDIA Technical Blog. Introducing NVFP4 for Efficient and Accurate Low-Precision Inference. 2025.
- NVIDIA CUTLASS docs: Blackwell SM100 GEMMs (docs.nvidia.com/cutlass/4.3.1/media/docs/cpp/blackwell_functionality.html)
- Colfax Research. CUTLASS Tutorial: Hardware-supported Block-scaling with NVIDIA Blackwell GPUs.
- NVIDIA HGX platform page (nvidia.com/en-us/data-center/hgx/); DGX B200 page.
- OpenAI. gpt-oss-120b & gpt-oss-20b Model Card. 2025. arXiv:2508.10925
- Xiao et al. SmoothQuant. ICML 2023. arXiv:2211.10438
- Ashkboos et al. QuaRot. NeurIPS 2024. arXiv:2404.00456
- Liu et al. SpinQuant. 2024. arXiv:2405.16406
- llama.cpp k-quants PR #1684 (github.com/ggml-org/llama.cpp/pull/1684)


---

# Literature notes: software kernels for weight-only / mixed-input quantized GEMM on GPUs

Lecture: "Compression Is Part of the Compute". Slice: what makes mixed-input (low-bit weight x FP16/BF16 activation) GEMM kernels fast or slow.

Confidence tags:
- **[V]** verified in the primary source's full text, README or table (usually by grepping the downloaded HTML or markdown, so the wording is exact)
- **[A]** abstract or summary level only
- **[I]** my own inference or arithmetic, not stated by the source
- **[F]** read off a figure or caption without exact numbers

Downloaded copies of the sources are in `../src/` next to this folder: marlin.txt, qserve.txt, fp6.txt, elephants.txt, kquants_pr.md, iq3s_pr.md, aqlm_readme.md, autoawq.md, gptq_readme.md, exl2_readme.md, hf_overview.md, machete.html.

---

## 0. The one-paragraph thesis support

The evidence below repeatedly supports three points:

1. **Bytes saved only turn into speed when the kernel reads the compressed format directly, at full bandwidth, and hides the decode behind tensor-core math.**
2. **Where that fails, a lower-bit format can be slower than a higher-bit one, and even slower than FP16.** Causes include a separate dequantize kernel, a codebook that spills out of L1, CUDA-core work in the main loop, and misaligned bit widths.
3. **Every fast kernel changes the memory layout offline to fit the hardware.** Examples are Marlin's reshuffling, Machete's prepacking, FP6-LLM's bit-level pre-packing, FLUTE's restructuring, QServe's compute-aware reordering, AWQ/TinyChat's SIMD-aware packing and FasterTransformer's interleaving.

The format, the layout and the kernel are designed together.

---

## 1. The ancestor: FasterTransformer's fast int→fp16 conversion ("magic number" trick)

**Source.** Young Jin Kim, Rawn Henry, Raffy Fahim, Hany Hassan Awadalla. "Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production." 2022. arXiv:2211.10017. https://arxiv.org/abs/2211.10017

**Fused vs separate dequantize [V].** They first rejected a separate dequantize kernel because it "would actually increase the amount of memory traffic as we would add a read of W and a write to Wdq". They fused dequantization into a CUTLASS GEMM instead. They then found that "the conversion from int to float … was slower than anticipated". They replaced the native I2F instruction with "a series of high throughput ALU and FP16 instructions".

**The trick [V].**
- An FP16 number X with 1024 ≤ X < 2048 has the integer X−1024 sitting directly in its mantissa bits.
- `0x6400 | Y` therefore builds the FP16 value Y+1024.
- For int8, subtract [1152, 1152]. That is 1024 plus 128, and the 128 undoes the unsigned bias.
- Two FP16 values fit in one 32-bit register, so conversion happens two at a time.
- For int4 they reorder each group of 8 nibbles offline, `[e0..e7] → [e0,e2,e4,e6,e1,e3,e5,e7]`, "to reduce the number of logic instructions". They then subtract [1032, 1032].
- This is the origin of the offline weight-interleaving plus bit-trick pattern that Marlin cites.

**Numbers [V], Table 1.** MoE GEMM on a V100, A = m×1024, B = 1024×4096, 40 tokens in total. Throughput is normalized to FP16.

| Active experts | Int8 with native I2F | Int8 with optimized I2F | Int4 with optimized I2F |
|---|---|---|---|
| 1 | 1.05 | 1.28 | 1.24 |
| 32 | 1.46 | 1.59 | 1.85 |
| Geomean | 1.26 | 1.35 | 1.56 |

- **Lecture nugget [V]:** with 1 active expert, int4 (1.24×) was slightly slower than int8 (1.28×). Four times fewer weight bits than FP16 gave only a 1.24–1.85× speedup.
- The conversion routine alone moved int8 from a geomean of 1.26× to 1.35×.

## 2. Google's CUTLASS mixed-input work on Ampere

**Source.** Manish Gupta (Google Research). "Mixed-input matrix multiplication performance optimizations." Google Research Blog, 26 Jan 2024. https://research.google/blog/mixed-input-matrix-multiplication-performance-optimizations/

**The two challenges [V, summary-level]:**
- **Data type conversion.** `mma` needs both operands in the same type.
- **Layout conformance.** An 8-bit operand's fragment layout across the warp's registers differs from the F16 fragment layout. After upcasting, each thread holds the wrong elements.

**Their solutions [A/V, summary-level]:**
- `FastNumericArrayConvertor` converts 4×U8 packed in a 32-bit register using `prmt` byte-permutes and the 0x6400 packed-FP16 arithmetic. It is reported as about 1.6× faster than naive conversion (about 6 vs 10 operations).
- `FragmentShuffler` uses `ldmatrix` in the 8-bit layout, then warp `shfl.sync` shuffles to fix the layout at runtime. This deliberately avoids offline global-memory preprocessing.
- Result on A100: "performance only slightly below or on par with" the same-type mixed-precision GEMM.

**Lecture contrast [I].** Google fixes the layout at runtime with shuffles. Marlin, Machete and FP6-LLM move that cost offline by storing the weights already in fragment order. Both routes exist to fix the same mismatch between the format and the hardware.

## 3. Marlin (and Sparse-Marlin)

**Source.** Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, Dan Alistarh. "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models." arXiv:2408.11743, Aug 2024 (the published version appeared at PPoPP 2025; the venue is not verified here).
- Paper: https://arxiv.org/abs/2408.11743
- Repo: https://github.com/IST-DASLab/marlin

### Headline claim
- **[V] Abstract:** "batchsizes up to 16-32 can be supported with close to maximum (4×) quantization speedup, and larger batchsizes up to 64-128 with gradually decreasing, but still significant, acceleration."
- **[V] Intro:** "MARLIN obtains speedups of approximately 3.9× relative to FP16 on an inference-optimized NVIDIA A10 GPU and large matrices, for batch sizes of up to 16-32 … Speedups gradually reduce, towards 1.5× at batch size 128, as the problem becomes compute-bound."
- **[V] Metadata caps the ideal speedup.** The ideal is 3.87×, not 4×: "note the 0.125 bits storage overhead of the group scales". An FP16 scale per 128 weights is 16/128 = 0.125 bpw, and 16/4.125 ≈ 3.88 [I arithmetic]. This is a direct lecture example of scale metadata eating into the speedup.
- **[V] Other kernels degrade with batch.** Other open-source 4-bit kernels "achieve relatively close to the optimal 3.87× speedup at batchsize 1 … their performance degrades quickly as the number of inputs is increased."
  - The comparison was on a 72k × 18k matrix on an A10 at group size 128.
  - The comparison set in Fig. 1 is ExLlamaV2, AWQ, bitsandbytes and PyTorch/torch-int4 [F].
- **[V] README framing:** close to 4× "up to batchsizes of 16-32 tokens (in contrast to the 1-2 tokens of prior work with comparable speedup)."

### Why speedup is theoretically available up to about 50 tokens [V]
- "an A10 GPU has a FLOPs/Bytes ratio of ≈200 … processing one input token takes 2 FLOPs per weight and the GPU can execute 100 FLOPs in the time it takes to load one 4-bit weight. Hence, memory loading will dominate runtime as long as the input batchsize is less than b_opt≈50."
- README version: "Most modern GPUs feature FLOP to byte ratios of around 100-200. Hence, as long as we perform less than 25-50 (tensor core) multiply-accumulates per 4-bit quantized weight, it should (theoretically) be possible to maintain near ideal 4x speedup."
- The hard part [V]: the kernel must "fully utilize all available GPU resources (global memory, L2 cache, shared memory, tensor cores, vector cores), *simultaneously*."

### Mechanisms [V unless noted]

**Offline reshuffling into tensor-core fragment order.**
- "reorganize weights such that the 16-byte vector read by each thread contains precisely its necessary 8 quantized weights of 4 separate 16×16 Tensor Core blocks."
- "within an INT32, weights are stored interleaved, according to the pattern 64207531, to power the … parallel decoding."
- 16×64 tiles are laid out contiguously in global memory.
- Group scales are also reshuffled offline (README).
- An XOR-swizzled shared-memory layout makes `ldmatrix` for the activations conflict-free.

**Fast INT4→FP16 conversion,** a "modified version of the binary manipulations of Kim et al. (2022)":
1. AND with a mask to isolate the nibble, and OR bits 1–7 to 0110010. This is "a single lop3 instruction, which we however seemingly need to emit explicitly".
2. This yields an FP16 number with exponent 50 whose low mantissa bits are the nibble.
3. Subtracting the FP16 value with exponent 50 and mantissa 0 gives the unsigned value.
4. "To make this value signed, we further have to subtract 8, which we can however fuse" into the same subtraction.
5. Two INT4s are decoded per INT32 using packed half2 math.

(Note: one WebFetch summary misreported this constant as "88". Grepping the paper text confirms it is 8.)

**Asynchronous global→shared pipeline.**
- `cp.async`, prefetching P−1 steps ahead with pipeline depth P = 4, plus register double-buffering from shared memory.
- P = 4 was chosen because it hides latency, fits in shared memory at M = 64, and is even, which lets the loop fully unroll so all shared-memory addresses become static.

**Cache policy.**
- Weights (B) are read exactly once, so they use "cp.async … with an evict_first cache-hint" to avoid evicting activations (A) from L2.
- "all activations are essentially always fetched from L2 cache and are further reused several times within registers."

**Instruction ordering.**
- Accumulation runs column-wise with m16n8k16 `mma.sync`, "so that we can pipeline the dequantization of the next B operand with the Tensor Core math of the current column."

**Striped partitioning.**
- An SM's tiles may span multiple column slices. This keeps all SMs busy on realistic shapes while minimizing global reductions.
- Reductions happen in the output buffer, which stays in L2, with FP32 accumulators temporarily downcast to FP16.

**Sustained clocks.**
- "reduced clock speeds significantly harm the relative speedups of prior kernels, but have no effect on MARLIN's virtually optimal performance (relative to the lower clock setting)."
- The lecture point: prior kernels were partly compute- or issue-bound on dequantization, so they suffered when the clock dropped [I interpretation].

### Why speedup decays at larger batch [V]
- "batchsizes smaller than 64 are memory-bound, while the larger batchsizes are compute-bound." This comes from a roofline over A10 layer shapes.
- As batch grows, the FP16 baseline also becomes compute-bound. Weight bytes stop mattering, and a mixed-input kernel can at best tie FP16 tensor-core throughput [I].

**End-to-end in vLLM [V], Table 2.** Speedup over vLLM FP16 at each batch size.

| Model | GPU × count | 1 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| Llama-2-7B | A10 ×1 | 2.93 | 2.74 | 2.26 | 1.78 | 1.20 |
| Llama-2-7B | 3090 ×1 | 2.69 | 2.30 | 1.84 | 1.28 | 1.11 |
| Llama-2-70B | A100 ×8 | 1.38 | 1.44 | 1.19 | 1.04 | 1.07 |
| Falcon-180B | A100 ×8 | 1.76 | 1.70 | 1.65 | 1.29 | 1.08 |

- "The largest speedups happen when inference is memory-bound (up to batchsize ≈16) and the GPUs are weaker or fewer in number."
- Serving benchmark (Llama-2-7B, RTX A6000): "MARLIN achieves approximately 2.8× latency reduction, while Sparse-MARLIN provides about 3.3×" (TPOT).

### Sparse-Marlin [V]
- It uses the 2:4 sparse tensor-core format: a values array plus a metadata array giving each non-zero's position within its group of 4.
- "additional speedups of up to 65% relative to the original (dense) variant" per layer.
- End to end it gives "an additional 1.2× end-to-end speedup on top of MARLIN" (Llama-2-7B, A10).
- Lecture framing [I]: sparsity is another compressed format that the hardware can only use with its own metadata layout.

## 4. Machete (Neural Magic / Red Hat, vLLM) and CUTLASS 3.x mixed-input on Hopper

**Source.** Lucas Wilkinson (Neural Magic). "Introducing Machete, a mixed-input GEMM kernel optimized for NVIDIA Hopper GPUs." Oct 14, 2024 (page last updated Mar 26, 2025). https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel (neuralmagic.com redirects here)

### Why Hopper needed a new kernel [V]
- "current state-of-the-art mixed-input linear kernels struggle in compute-bound scenarios". Marlin, gemlite and fbgemm_i4 were benchmarked on an H100 [F].
- "For Marlin, the poor performance on Hopper architecture primarily stems from the use of outdated 'mma' tensor core operations … Using only 'mma' instructions results in a loss of approximately 37% of peak compute throughput."
- "Marlin's weight pre-shuffling, being hand-derived and implemented, makes it challenging to easily adapt to the new 'wgmma' layouts."
- Machete is built on CUTLASS 3.5.1, starting from example 55.

### The prepack / pre-shuffle idea [V]
- The blog calls pre-shuffling "an important previously undiscussed optimization used by both Marlin and Machete".
- The problem: `ldmatrix` "isn't available for 4-bit types. When working with 4-bit elements and using float16 or bfloat16 as the compute type … we would naively need to resort to performing four 8-bit shared memory loads per tensor core operation."
- The fix: reorder the weights offline so that each thread's elements are contiguous. This allows one 32-bit shared-memory load, and interleaving four 8×8 tiles allows 128-bit loads.
- Machete "employ[s] CUTLASS CUTE layout algebra to construct a description of the repacked layout for a full weight matrix using instruction layout definitions".
  - The layout is derived from the instruction's layout rather than hand-coded, so it can follow new instructions.
  - Lecture point: the storage format is defined by the tensor-core instruction [I].
- Upconversion: nibbles are interleaved so that "multiple nibbles can be efficiently extracted and up-converted in parallel … shifting the nibbles into the lower four bits of their destination registers using simple bit shifts and masking operations and then expanding in-place."

### Numbers [V text, baseline read from figures]
- "At batch sizes of 128 and above, the performance is competitive with FP16, meaning there is no longer a trade-off between prefill performance or high-batch size performance and improved low-batch and decode performance."
  - Note: the best case in the compute-bound regime is *tie* FP16, not beat it.
- Llama 3.1 70B, w4a16, 1×H100: "In the higher user requests rates (3+ req/s), we see a geomean speedup of 29% for input token throughput and 32% for output token throughput."
- Llama 3.1 405B, 4×H100: "geomean speedup of 42% for both input token and output token throughput."
- **Caution:** the text does not name the baseline. The figures compare against the other mixed-input kernels, most likely Marlin as vLLM's previous default [I].
- Stated future work includes "Optimizing low batch size performance … (sub-32)". This implies Machete was not yet the best choice at small batch sizes [I].

### CUTLASS example 55 README (NVIDIA) [V]
https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm

- The narrow type is "always pass[ed] … through the register file" and upcast there.
- "it's recommended to reorder the narrow data type tensor such that elements read into register file by the same thread are contiguous in global and shared memory" (helpers `compute_memory_reordering_atom`, `reorder_tensor`).
- "The scale only mode for `fp8 x int4` is significantly slower than direct conversion mode. There is a lookup-table workaround … However, it requires modifications to the encoding of quantized weights and scale factors."
  - This is a vendor saying outright that the scale encoding has to be changed so the kernel can run fast.
- The README says it is expected to be performant "for problems that are compute bound". Memory-bound cases are listed as "currently optimizing".

## 5. QServe / QoQ (W4A8KV4)

**Source.** Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han. "QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving." arXiv:2405.04532 (v1 May 2024; v3 May 2025, MLSys 2025). https://arxiv.org/abs/2405.04532 . Code: https://github.com/mit-han-lab/omniserve

### Core argument [V]
- "existing INT4 quantization methods suffer from significant runtime overhead (20-90%) when dequantizing either weights or partial sums on GPUs."
- "The key insight driving QServe is that the efficiency of LLM serving on GPUs is critically influenced by operations on low-throughput CUDA cores."
- **Quotable:** "On data center GPUs like A100, a CUDA core operation is as expensive as 50 INT4 tensor core operations. Therefore, reducing bit precision not necessarily speeds up LLM inference."

### Main-loop overhead (Section 3.2, "Why Not W4A4KV4") [V]
- In an m×n×k GEMM, k is the sequential main loop. "In LLM serving, m is small and n,k are large. Thus, the main loop is long."
- "W8A8 is fast because its main loop only contains tensor core operations and all dequantization operations are present in the epilogue. Atom-W4A4 and TensorRT-LLM-W4A16 suffer from significant partial sum or weight dequantization overhead in the main loop."

**Why W4A4 is slow.**
- Accuracy needs per-group scales. Per-group scales force INT32→FP32 dequantization of partial sums inside the main loop, on CUDA cores.
- "the peak performance of FP32 CUDA cores is merely 2% of their INT4 tensor core counterparts … de-quantizing one single partial sum in Atom is equivalent to 50 tensor core MACs."
- Atom also needs two sets of registers for partial sums (FP32 and INT32).
- "the state-of-the-art W4A4 serving system, Atom, exhibits 20-25% lower performance than its W4A16 and W8A8 counterpart in TensorRT-LLM." Despite a 2× higher theoretical peak, W4A4 systems "significantly lag behind TRT-LLM-W8A8."
- Fig. 18 caption: Atom-W4A4 dequantization overhead is "up to 90%".

### Roofline crossover (A100: 312/624/1248 TOPS for FP16/INT8/INT4, 2 TB/s) [V]
- "W4A16 has a higher theoretical throughput when m<78, while W8A8 performs better when m>78."
- Below that point the GEMM is memory-bound and weight traffic dominates. Above it the GEMM is compute-bound, and W8A8 wins on INT8 tensor-core throughput.

### Mechanisms [V]
- **Progressive group quantization.** Weights are first quantized per-channel symmetric to INT8, then per-group asymmetric to INT4. The main loop therefore dequantizes INT4 to INT8 only and runs on INT8 tensor cores.
  - A protective range of [−119, 119] keeps intermediates in the INT8 range with no saturation instructions.
- **Register-level parallelism.** "4-way register-level parallelism" decodes four INT4 at once, using `vadd4`-style INT8 SIMD.
- **Subtraction after multiplication.** Zero-point handling moves to the epilogue, where it fuses with the scales as in W8A8.
- **Compute-aware weight reordering.** Weights are stored in the order they are consumed, which enables 128-bit loads per thread and removes pointer arithmetic that would otherwise run on CUDA cores.
- **SmoothAttention.** Keys have fixed outlier channels (values do not). Keys are scaled per channel by λ_i = max(|K_i|)^α with α = 0.5 [V]. The inverse scale is folded into the preceding query/key projection weights, which is my recollection of the paper [I]. The result is that KV4 stays accurate.
- Fig. 18: QServe's dequantization overhead is "comparable with TRT-LLM-W4A16, but since we perform computation on INT8 tensor cores, we enjoy 2× higher throughput."

### Throughput results [V]
- Llama-3-8B: 1.2× on A100 and 1.4× on L40S. Qwen1.5-72B: 2.4× on A100 and 3.5× on L40S. Both are relative to the best TensorRT-LLM configuration among FP16, W8A8 and W4A16.
- Ranges: 1.2–2.4× on A100, 1.5–3.5× on L40S, and 2.5–2.9× over Atom and QuaRot (W4A4) on A100.
- "QServe on L40S GPU can achieve even higher throughput than TensorRT-LLM on A100", which the paper describes as cutting dollar cost about 3×.
- Appendix Table 6, tokens/s vs TRT-LLM W8A8KV8 (GPU given by the paper's appendix setup):

| Model | TRT-LLM W8A8KV8 | QServe |
|---|---|---|
| Llama-3-8B | 2387.55 | 2980.69 |
| Llama-2-7B | 2339.97 | 2860.01 |
| Mistral-7B | 2427.64 | 3031.93 |

### A second lower-precision-is-slower example: KV4 attention [V]
- Simply replacing KV8 with KV4 in TRT-LLM's decode attention was "1.7× speedup on L40S, but results in 1.2× slowdown on A100 … the devil is in the slow CUDA cores."
- Table 1 gives naive KV4 at 0.86–0.90× the speed of 8-bit KV on A100. Their optimized version reaches 1.29–1.51×.

## 6. FP6-LLM / TC-FPx (6-bit weights, Microsoft DeepSpeed)

**Source.** Haojun Xia, Zhen Zheng, Xiaoxia Wu, Shiyang Chen, Zhewei Yao, Stephen Youn, Arash Bakhtiari, Michael Wyatt, Donglin Zhuang, Zhongzhu Zhou, Olatunji Ruwase, Yuxiong He, Shuaiwen Leon Song. "FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design." arXiv:2401.14112, Jan 2024. A later version appeared at USENIX ATC 2024 as "Quant-LLM" [I, not verified here]. https://arxiv.org/abs/2401.14112

### Why fused dequantization matters [V]
- In the dual-kernel approach, "the de-quantized FP16 weights will be written to GPU DRAM before being read by the second GPU kernel, resulting in 2× DRAM access."
- "such inference speed would be even slower than that of the model without quantization."

### The irregular bit-width problem [V]
- Shared memory has 32 banks, and each serves a 32-bit word.
- A thread needing 2×6 = 12 bits reads a whole word: "20 out of 32 bits (62.5%) unused."
- A pair of weights that straddles two words costs 64 bits read for 12 used, which is 81.25% wasted.
- Aligned-access rules make this worse.

### Ahead-of-time bit-level pre-packing [V]
- **Weight split:** "we split each weight into several segments, where the bit-width of each segment is 2^n, e.g. each 6-bit weight can be split into either 2+4 or 4+2." The 2-bit and 4-bit segment arrays are packed separately.
- **Step 1, per-thread gathering.** The warp tile is 64×64, split into 4 slices, each split into 4 chunks of 16×16. Each thread's weights are gathered in the order Tensor Cores will consume them.
- **Step 2, per-warp assembly.** Each thread's 32-bit words are combined "in a jagged order". A warp then reads consecutive 32-bit items, "fully avoiding bank conflict", and DRAM→shared copies are plain 128-byte blocks.
- "all the techniques … are independent of the actual bit-width."

### SIMT-efficient parallel dequantization [V]
- **Runtime weight stitching.** The 2-bit and 4-bit segments are recombined in registers.
- **Cheap cast via exponent-bias identity.** FPx→FP16 becomes a few bit operations: "two bit-wise 'and', one 'shifting', and one 'or'". The exponent-bias difference is folded into a multiply by 2^(bias_fp16 − bias_fpx), which merges with the scale.
- **4-way parallel dequantization.** Four FP6 values in one 32-bit register are dequantized at once, which cuts SIMT instructions about 4×.
- **Measured cost [V].** Arithmetic/logic unit (ALU) utilization rises "from 6.36% to 38.8% on average". Fused multiply-add (FMA) unit utilization rises from 0.33% to 16.64%. The paper calls this "strong evidence that the SIMT-efficient designs … are essential". The work is hidden by software-pipelining it against Tensor Core math.

### Numbers (A100-40GB, CUDA 11.8) [V]
Linear layers in the decode phase:
- TC-FPx beats bitsandbytes (W4A16), cuBLAS (W16A16) and TensorRT-LLM (W8A16) by up to 8.9×, 2.6× and 1.9× respectively.
- Averages at batch 8/16/32:
  - vs bitsandbytes: 7.6×/7.5×/6.6×
  - vs cuBLAS FP16: 2.2×/2.2×/2.0×
  - vs TRT-LLM W8A16: 1.3×/1.3×/1.2×

**The key "bigger can be faster" data point:**
- "BitsandBytes is constantly slower than cuBLAS, which is 29.6% as fast as cuBLAS on average … BitsandBytes adopted the dual-kernel method."
- So a 4-bit format ran about 3.4× slower than FP16 [I: 1/0.296]. Meanwhile 6-bit TC-FPx ran 2.2× faster than FP16. The difference is the kernel, not the bit count.
- Caveat [I]: this was bnb at commit f1ef74f at batch 8–32. bnb's later batch-1 inference kernel would not apply at these batch sizes anyway (see §8).

**Versus TensorRT-LLM 4-bit (LLaMA-65B layers):**
- TC-FPx W6A16, fine-grained W4A16 and coarse-grained W4A16 beat cuBLAS by up to 2.4×, 3.0× and 3.3×.
- "TC-FPx … is 1.06×/1.04×/0.94× faster than Fine-grained_W4A16 … at batch size 8/16/32." So **6-bit roughly matched, and at batch 8/16 slightly beat, group-wise 4-bit**, although it moves 1.5× more weight bits [I].
- It is "only 16% / 17% / 24% slower than" coarse-grained (per-row) W4A16.
- Lecture point [I]: fine-grained group scales carry a real decode cost. 4-bit with per-group scales loses about as much as 2 extra bits of payload.

**End-to-end:**
- LLaMA-70B on a single GPU: 1.69×–2.65× the normalized throughput of the FP16 baseline, which needs 2 GPUs.
- OPT-30B: 1.72×–4.05×.

**Crossover [V].** "the performance of our TC-FPx kernel, cuBLAS kernel, and TensorRT-LLM's W8A16 kernel will eventually converge … when the inference batch size is larger (bigger than 128), as their performance will all be bounded by the peak computing power of Tensor Cores."

**Parallel with FLUTE [V/I].**
- FLUTE (Guo et al., arXiv:2407.10960) splits 3-bit weights into "two partitions: one containing the 1-bit portion and the other the 2-bit portion". This is the same power-of-two bit-slicing idea as FP6-LLM's 2+4, applied to 3-bit plus a lookup table.
- FLUTE's abstract: "offline restructuring of the quantized weight matrix to minimize bit manipulations associated with unpacking, and vectorization and duplication of the lookup table to mitigate shared memory bandwidth constraints … At batch sizes < 32 and quantization group size of 128 … the FLUTE kernel can be 2-4x faster than existing GEMM kernels", with 1.5–2× end-to-end throughput.
- FLUTE also uses Stream-K partitioning for low-batch load balance [V, summary-level].

## 7. Block and group metadata: effective bits per weight and decode cost

### llama.cpp k-quants (GGUF)
**Source.** Iwan Kawrakow (ikawrakow), "k-quants", ggml-org/llama.cpp PR #1684, merged 2023-06-05. https://github.com/ggml-org/llama.cpp/pull/1684

**Formats [V].** Super-blocks have 256 weights. "Type-0" means w = d·q. "Type-1" means w = d·q + m.

| Type | Kind | Structure | Scale/min bits | Effective bpw |
|---|---|---|---|---|
| Q2_K | type-1 | 16 blocks × 16 weights | 4-bit scales and mins | **2.5625** |
| Q3_K | type-0 | 16 blocks × 16 | 6-bit scales | **3.4375** |
| Q4_K | type-1 | 8 blocks × 32 | 6-bit scales and mins | **4.5** |
| Q5_K | type-1 | as Q4_K | as Q4_K | **5.5** |
| Q6_K | type-0 | 16 blocks × 16 | 8-bit scales | **6.5625** |

- Q8_K is used for intermediate activations only.
- The metadata overhead is 0.44–0.56 bpw [I arithmetic from the table].

**Lower bits slower in the same PR's own table [V].** ms/token at 4 threads, 7B model:

| Measure | Q3_K_S (2.75G) | Q4_K_S (3.56G) |
|---|---|---|
| RTX-4080 | 18.6 | **15.5** |
| M2 Max CPU | 81 | **50** |

13B:
- RTX-4080: Q3_K_S 29.2 and Q3_K_M 29.3 vs Q4_K_S 26.2.
- M2 Max: Q3_K_M 148 vs Q4_K_S 95.
- Q2_K (2.67G) ran at 15.5 on the 4080, the same as Q4_K_S.

So **the 3-bit k-quant, 23% smaller, was about 20% slower on GPU and about 60% slower on the M2 CPU than 4-bit** [V numbers; I percentages]. The likely cause is the 3-bit unpack cost (the 2-bit plus 1-bit high-mask layout) [I, not stated in the PR]. Caveat: this is a mid-2023 implementation, and later llama.cpp versions may differ.

### llama.cpp i-quants: codebook decode cost
**Source.** ikawrakow, "IQ3_S: a much better alternative to Q3_K", PR #5676, 2024-02-23. https://github.com/ggml-org/llama.cpp/pull/5676

- **[V]** IQ3_S is 3.4375 bpw, "the exact same size as Q3_K", with a 40–70% smaller quantization error.
- Speed is "similar to Q3_K" on CUDA (RTX-4080), AVX2 and Metal.
- **[V] On the ARM CPU:** "Performance on the M2 Max CPU with ARM_NEON intrinsics is pathetic - only about 10 t/s for a 7B model compared to 22.5 t/s for Q3_K_S. The IQ series of quants use 'codebooks' to encode groups of 4 or 8 weights. For IQ3_S this requires 4 memory loads from a lookup table of 2048 bytes to setup one 128-bit SIMD register."
- The bits and the size are identical, and the decode is about 2.2× slower. The format's decode path decides speed [I].
- Bit accounting [V]: the 512-entry codebook versus 256 for IQ3_XXS costs 0.25 bpw, and not enforcing even sign parity costs 0.125 bpw.
- (This is CPU, not GPU. Use it as the cleanest "same bits, different decode" example.)

### bitsandbytes / QLoRA NF4
**Source.** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. "QLoRA: Efficient Finetuning of Quantized LLMs." 2023. arXiv:2305.14314.

- **[V] Double quantization:** "for a blocksize of 64, this quantization reduces the memory footprint per parameter from 32/64=0.5 bits, to 8/64+32/(64·256)=0.127 bits." So NF4 costs about 4.5 bpw without double quantization and about 4.127 bpw with it [I arithmetic].
- **[V] Compute path:** "we dequantize the tensor to BFloat16, and then perform a matrix multiplication in 16-bit."
- **[V] bitsandbytes 0.40.0 release notes** (https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.40.0):
  - Batch-1 4-bit inference kernel speedups vs 16-bit for inner dimension ≥ 4096: "2.2x for Turing … 3.4x for Ampere … 4.0x for Ada/Hopper".
  - "The inference kernels for batch size 1 are about 8x faster than 4-bit training kernel for QLoRA."
  - Users can "separate a multi-batch 4-bit query into multiple requests with batch size 1". In other words, the fast path existed only at batch 1.
- **[V] Hugging Face blog** (Younes Belkada et al., "Overview of natively supported quantization schemes in 🤗 Transformers", 12 Sep 2023, https://huggingface.co/blog/overview-quantization-transformers):
  - "bitsandbytes 4-bit models are slow compared to GPTQ when using generate."
  - With batch_size=4 and use_cache=True on an A100, GPTQ (exllama kernels) was "twice as fast".
- See §6 for FP6-LLM's measurement that bnb W4A16 ran at 29.6% of cuBLAS FP16 speed.

### ExLlamaV2 / EXL2
**Source.** turboderp, ExLlamaV2 README. https://github.com/turboderp-org/exllamav2 (archived; development continues in ExLlamaV3)

- **[V] Format:** "supports 2, 3, 4, 5, 6 and 8-bit quantization. The format allows for mixing quantization levels within a model to achieve any average bitrate between 2 and 8 bits per weight."
  - It can also mix levels within a layer: "more important weights (columns) are quantized with more bits."
  - "The same remapping trick that lets ExLlama work efficiently with act-order models allows this mixing of formats to happen with little to no impact on performance."
  - Parameters are chosen by quantizing each matrix several ways and minimizing the maximum error at the target bitrate.
- **[V]** Llama2-70B runs at 2.55 bpw on one 24 GB GPU.
- **[V] README decode speeds, Llama2-7B on RTX 4090:**

| Bitrate | Speed |
|---|---|
| 3.0 bpw | 257 t/s |
| 4.0 bpw | 211 t/s |
| 5.0 bpw | 179 t/s |

- This is a counterexample where lower bits are faster, which shows a kernel built for the format can scale. The scaling is sub-linear, though: 1.67× fewer bits gives 1.44× more speed [I arithmetic].
- The README calls these "quick tests" at batch 1, and the numbers depend on version.

## 8. AWQ / TinyChat, the original GPTQ kernel, AutoAWQ GEMM vs GEMV

### AWQ / TinyChat
**Source.** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han. "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration." arXiv:2306.00978 (MLSys 2024 best paper).

- **[V, summary-level]** "We avoid writing dequantized weights into DRAM by fusing dequantization kernels with the matrix multiplication kernel."
- **SIMD-aware weight packing for ARM NEON (128-bit).** Weights are reordered as w0, w16, w1, w17, …, w15, w31, so that "just three SIMD instructions to unpack all 32 weights, as opposed to 3 scalar instructions per weight in conventional packing."
- Kernel fusion of layernorm, QKV and positional embedding.
- "3.2-3.3× average speedup compared to the FP16 implementation by Huggingface."
- 30 tokens/s on a laptop RTX 4070 with 8 GB.

### AutoAWQ README (casper-hansen/AutoAWQ; now deprecated and adopted by vLLM llm-compressor)
Setup: RTX 4090, AutoAWQ 0.1.6. [V]

- GEMV kernel: "20% faster than GEMM, only batch size 1 (not good for large context)."
- GEMM kernel: "Much faster than FP16 at batch sizes below 8."
- "In the scenario of being compute-bound, which happens at higher batch sizes, you will not gain a speed-up using a W4A16 quantized model because the overhead of dequantization will slow down the overall generation."

**Same weights, different kernel [V].** Mistral-7B, batch 1, prefill 2048 tokens:

| Kernel | Prefill | Decode |
|---|---|---|
| GEMM | **3897 tok/s** | 114 t/s |
| GEMV | **904 tok/s** | 131 t/s |

- At batch 8 the GEMV kernel falls to 486 decode tok/s against GEMM's 1185, and the README marks it "avoid using".
- So the right kernel depends on the phase: GEMV wins decode at batch 1, and GEMM wins prefill by about 4.3× [I arithmetic].

### GPTQ's original kernel
**Source.** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv:2210.17323 (ICLR 2023). Repo: https://github.com/IST-DASLab/gptq

- **[V, summary-level]** A "quantized-matrix full-precision-vector product" kernel that dequantizes on the fly, batch size 1 only. Activations stay FP16.
- **[V, summary-level]** OPT-175B at 3-bit:
  - A100: 230 ms → 71 ms per token (3.25×)
  - A6000: 589 ms → 130 ms (4.5×)
  - Caveat [I]: the FP16 baseline is split across 5×A100 or 8×A6000, so part of the gain is avoiding multi-GPU communication.
- **[V] README, kernel alone:** "Optimized 3bit kernels, which are considerably faster especially on the A100, e.g. 1.9x -> 3.25x generation speedup for OPT-175B." Same 3-bit format, and the kernel rewrite nearly doubled the gain.
- **[V] README:** "our 3-bit kernels are currently only optimized for OPT-175B running on 1xA100 or 2xA6000 and may thus yield suboptimal performance on smaller models or on other GPUs."

## 9. Vector / codebook quantization: when decode cost dominates

### AQLM
**Source.** Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh. "Extreme Compression of Large Language Models via Additive Quantization." arXiv:2401.06118 (ICML 2024). Repo: https://github.com/Vahe1994/AQLM

**README kernel table [V].** All of these are about 2-bit formats.

| Kernel | Codebooks | Speedup (as stated) |
|---|---|---|
| Triton, generic K×N | K×N | "Up to ~0.7x", i.e. slower than FP16 |
| CUDA 1x16 (best accuracy) | one 16-bit codebook | "Up to ~1.3x" |
| CUDA 2x8 ("OK" accuracy) | two 8-bit codebooks | "Up to ~3.0x" |
| Numba, CPU | K×8 | "Up to ~4.0x" |

- Paper [V, summary-level]: on an RTX 3090, 1×16 gives about 1.2–1.31× layer speedups, and 2×8 gives 1.57–3.05×.
- "multiple smaller codebooks allow efficient GPU cache utilization, leading to greater speedup, at the price of slightly lower accuracy."
- Codebook size [I arithmetic, assuming group dimension 8 and FP16 entries]: 1×16 means 2^16 × 8 × 2 B = 1 MiB, far larger than L1. 2×8 means 2 × 256 × 8 × 2 B = 8 KiB.

**QuIP# measurement of AQLM (Table 6) [V].** RTX 4090, HF Llama implementation, decode tok/s:

| Method | Llama-2-7B | Llama-2-70B |
|---|---|---|
| FP16 | 33.1 | OOM |
| AQLM 2-bit | **20.6** | 8.27 |
| QuIP# 2-bit | 106.3 | 25.9 |

- Caption: "Unlike AQLM, whose codebook is too large to fit in L1 cache, QuIP# achieves significant speedups over FP16."
- **So 2-bit AQLM ran about 38% slower than FP16 with 8× fewer weight bytes.** This is a cited example for the lecture [V numbers; I %].

### QuIP#
**Source.** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." arXiv:2402.04396 (ICML 2024).

- **[V] E8P codebook.** It has 2^16 entries but is stored as a 256-entry table plus signs and shifts: 8 bits of lookup, 7 bits of signs and 1 bit of shift. "E8P requires only 1KiB of space and therefore fits in the L1 cache of any modern GPU."
- **[V] Randomized Hadamard transform** at inference: Θ(n log n), with no floating-point multiplies.
- **[V] Table 5,** RTX 4090, FlashAttention Llama, batch 1:

| Model | 2-bit | 4-bit |
|---|---|---|
| Llama-2-7B | 170.50 tok/s (29.60% of peak memory bandwidth) | 117.73 tok/s (40.87%) |
| Llama-2-70B | 32.74 tok/s (56.84%) | OOM |

- Even a codebook built for decode reaches only 30–57% of memory bandwidth at 2 bits. Decode compute, not bytes, sets the ceiling for small models [I]. And 2-bit is only 1.45× faster than 4-bit at 7B [I arithmetic].

### QTIP
**Source.** Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa. "QTIP: Quantization with Trellises and Incoherence Processing." arXiv:2406.11235 (NeurIPS 2024).

- **[V, summary-level]** Unstructured vector quantization "requires exponential time and space in both the bitrate and dimension". Codebooks must fit in L1, which caps dimension at about 8.
- QTIP uses a "bitshift trellis" and compute-based codes:
  - 1MAD: about 2 instructions plus `vabsdiff4`
  - 3INST: 3 ALU instructions
  - HYB: about 2 instructions per weight amortized, with a 2 KB table
- These replace lookups with arithmetic, so the format is designed around the decode instruction count.
- Batch-1 decode on an RTX 6000 Ada:
  - Llama-2-7B: FP16 55.9, QTIP 2-bit 188, QuIP# 2-bit 186 tok/s
  - Llama-2-70B at 2-bit: QTIP 23.5 vs AQLM 8.78 tok/s
- No batch > 1 results.

## 10. When weight-only quantization stops helping (batch and arithmetic-intensity crossover; prefill vs decode)

**The mechanism** [V: Marlin, QServe; I: synthesis].
- A GEMM with m activation rows does 2m FLOPs per weight.
- FP16 weights move 2 bytes per weight, so arithmetic intensity is m FLOP/byte.
- 4-bit weights move about 0.5 bytes (plus scales), so intensity is about 4m FLOP/byte.
- Once m is large enough that even FP16 is compute-bound, weight bytes no longer matter. Dequantization then becomes pure added work on the same SMs, and it competes with tensor-core issue.

**Crossover numbers.**

| Source | Crossover | Tag |
|---|---|---|
| Marlin, A10 (≈200 FLOP/B) | b_opt ≈ 50; "batchsizes smaller than 64 are memory-bound, while the larger batchsizes are compute-bound"; end-to-end speedup 1.20× (7B) at batch 128 and 1.04× (70B on 8×A100) at batch 64 | [V] |
| QServe, A100 roofline | W4A16 beats W8A8 only for m < 78 | [V] |
| FP6-LLM | W6A16, FP16 and W8A16 kernels "converge" above batch 128 | [V] |
| Machete, H100 | the best Hopper mixed-input kernel is only "competitive with FP16" at batch or prefill length ≥ 128 | [V] |
| Hopper `mma` vs `wgmma` | a kernel stuck on the old instruction loses about 37% of peak | [V] |
| AutoAWQ | "compute-bound … you will not gain a speed-up using a W4A16 quantized model because the overhead of dequantization will slow down the overall generation" | [V] |

- Marlin's A100 speedups are smaller than on the 3090 because of the A100's higher bandwidth and compute: "overheads such as pipeline startup latency or suboptimal partitioning relatively bigger" [V].
- On the crossover arithmetic [I]: with the A100's 312 TFLOPS and 2 TB/s (156 FLOP/B), FP16 weights become compute-bound around m ≈ 156. Pure 4-bit loads alone would hit the ridge around m ≈ 39. Past that point the 4-bit kernel's advantage shrinks, reaching zero by about m ≈ 156.

**Prefill is where the dequant tax shows [V].** HF blog, A100-80GB, Llama-2-13B, prompt length 512, prefill only:

| Precision | Batch 1 | Batch 16 |
|---|---|---|
| FP16 | 27.1 tok/s | **228.8 tok/s** |
| GPTQ 4-bit (exllama) | 29.7 tok/s | 167.7 tok/s |
| bitsandbytes 4-bit | 19.2 tok/s | 140.4 tok/s |

At batch 16 with 512-token prompts, **both 4-bit formats are 27–39% slower than FP16** [I %]. Quantization gives capacity here, not speed.

**Decode vs prefill in serving [I].** Decode at small or moderate batch is memory-bound, which is where W4A16 wins. Prefill, and decode at large batch, are compute-bound, where W4A16 at best ties FP16 and can lose. This is why production stacks keep several kernels:
- vLLM keeps Marlin for Ampere and Machete for Hopper.
- AutoAWQ keeps both GEMM and GEMV.
- QServe moves to W4A8 so the compute-bound regime also gets faster (INT8 tensor cores).

---

## 11. Quick "lower bits were slower" examples, ranked by strength

| # | Example | Source | Tag |
|---|---|---|---|
| 1 | AQLM 2-bit: 20.6 tok/s vs FP16 33.1 tok/s (Llama-2-7B, RTX 4090, HF). The codebook does not fit in L1. The AQLM README itself lists the generic Triton kernel at "up to ~0.7x". | QuIP# Table 6; AQLM README | [V] |
| 2 | bitsandbytes 4-bit ran at 29.6% of cuBLAS FP16 speed (dual kernel), while 6-bit TC-FPx ran 2.2× faster than FP16 on the same A100 layers. | FP6-LLM | [V] |
| 3 | 6-bit TC-FPx was 1.06×/1.04× *faster* than TensorRT-LLM group-wise 4-bit at batch 8/16. | FP6-LLM | [V] |
| 4 | llama.cpp Q3_K_S (2.75 GB) at 18.6 ms/tok vs Q4_K_S (3.56 GB) at 15.5 ms/tok on an RTX 4080, and 81 vs 50 ms on the M2 Max CPU. | k-quants PR #1684 | [V] |
| 5 | Same 3.4375 bpw, IQ3_S at about 10 t/s vs Q3_K_S at 22.5 t/s on the M2 Max CPU, because of codebook lookups. | PR #5676 | [V] |
| 6 | W4A4 (Atom) was 20–25% slower than TRT-LLM W4A16 and W8A8 despite 2× the peak. "reducing bit precision not necessarily speeds up LLM inference." | QServe | [V] |
| 7 | Naive KV4 attention was 1.2× slower than KV8 on A100. | QServe | [V] |
| 8 | Int4 at 1.24× vs int8 at 1.28× over FP16 for an MoE GEMM with 1 active expert, V100. | Kim et al. 2022 | [V] |
| 9 | Prefill at batch 16: GPTQ-4bit and bnb-4bit ran at 73% and 61% of FP16 throughput. | HF blog | [V; I %] |
| 10 | Same format, different kernel: the GPTQ 3-bit kernel rewrite went from 1.9× to 3.25×, and AutoAWQ's GEMV kernel prefilled 4.3× slower than its GEMM kernel. | GPTQ README; AutoAWQ README | [V] |

Counterexample to cite for balance [V]: ExLlamaV2 EXL2 gets faster monotonically at lower bpw at batch 1 (3.0 bpw at 257 t/s, 4.0 at 211, 5.0 at 179, 4090, Llama2-7B), because its kernel was built for the format.

## 12. Not verified / gaps
- Machete's 29/32/42% baseline is not named in the text. It is presumably the other mixed-input kernels, likely Marlin.
- I did not find a primary source showing an ExLlamaV2 3-bit or 2-bit kernel slower than 4-bit on GPU. The README data shows the opposite.
- The Marlin paper's Fig. 1 per-kernel numbers for the competitors were not extracted; only the qualitative "degrades quickly" is verified.
- No vLLM documentation was found stating that AQLM switches to dequantize-then-matmul for large batches.
- The venues (PPoPP 2025 for Marlin, ATC 2024 for FP6-LLM/Quant-LLM, MLSys 2025 for QServe) come from memory and are not verified here.


---

# Literature notes: FLUTE and lookup-table (LUT) quantized matmul kernels

For the lecture "Compression Is Part of the Compute".
Compiled 2026-09-25 from arXiv HTML full texts (downloaded and read locally) and official GitHub READMEs/source.

Confidence tags:
- **[V]** verified in the paper's full text (or the named README or source file)
- **[A]** from the abstract only
- **[I]** my inference or arithmetic, not stated in the source

---

## 0. The frame: two kinds of "LUT kernel"

The literature uses "LUT" for two opposite ideas. The lecture should keep them apart.

| | **LUT over weight values** (dequantize by table) | **LUT over activation partial sums** (compute by table) |
|---|---|---|
| What the table holds | The real value each weight code stands for (e.g. the 16 NF4 levels) | Precomputed sums of activations for every possible pattern of weight bits (e.g. all 16 signed sums of 4 activations) |
| Built when | Offline; fixed per tensor (scaled per group) | Online, for every new activation vector or tile |
| After the lookup | An FP16 weight goes into a normal FP16 **tensor-core MMA** | The looked-up partial sums are added up. No multiply and no dequantization. |
| Examples | FLUTE, SqueezeLLM kernels, bitsandbytes NF4, QTIP HYB, AQLM (codebooks) | LUT-GEMM (GPU), T-MAC (CPU), LUT Tensor Core (proposed hardware) |
| Best fit | GPUs, because tensor cores do the math | CPUs with byte-shuffle instructions (TBL/PSHUF), or new hardware |

FLUTE says this directly [V]: LUT-based matmuls such as LUT-GEMM "cannot leverage specialized accelerators on modern GPUs which are typically optimized for FP matmuls. We thus seek efficient kernels which can simultaneously make use of quantized representations (to minimize memory movement) as well as GPU-native matrix multiplications in FP."

---

## 1. FLUTE

**Citation:** Han Guo, William Brandon, Radostin Cholakov, Jonathan Ragan-Kelley, Eric P. Xing, Yoon Kim. "Fast Matrix Multiplications for Lookup Table-Quantized LLMs." Findings of EMNLP 2024. arXiv:2407.10960 (v4, 17 Jan 2025). Code: https://github.com/HanGuo97/flute. The EMNLP Findings venue is confirmed by the README news entry of Oct 5, 2024 and its BibTeX [V].
FLUTE stands for "Flexible Lookup Table Engine."

### 1.1 Problem it targets
- Small-batch LLM decoding is memory-bound. "the primary bottleneck is the cost of transferring model parameters from the GPU's global memory to its registers" [V].
- The paper's roofline example for A100-80GB [V]: about 3×10^14 FLOP/s of FP16 matmul against about 1.5×10^12 B/s of DRAM bandwidth. Any kernel doing fewer than about 200 FLOPs per byte is bandwidth-bound.
- A fused weight-only kernel must do four steps [V]:
  1. Move the quantized weights from DRAM to SRAM.
  2. Dequantize them on chip.
  3. Run the FP matmul.
  4. Write the result back.
- Existing kernels (bitsandbytes, Marlin, BitBLAS) show "up to four times faster when going from W16A16 to W4A16". They are "typically specialized to 4-bit quantization", and the LUT-capable ones "are generally slower than the uniform counterparts." bitsandbytes and BitBLAS "do not allow for 3 bit-quantized weights" [V].
- The paper names three challenges [V]:
  1. Packing and unpacking sub-8-bit data, especially odd widths, so that it matches GPU-native matmul formats.
  2. LUT dequantization "involves dynamic indexing". GPUs "do not natively support dynamic indexing of a lookup table in their fastest on-chip registers". Uniform INT→FP conversion can instead use "assembly-level optimizations... through bit-level manipulations."
  3. Standard tile partitioning becomes inefficient "with small batches and low bit-width weights."
- Setting [V]: weight-only, non-uniform/LUT quantization at 4 and 3 bits (2-bit only when built from source, per the README), with group sizes 32, 64, 128 and 256. Main focus: batch sizes below 32, group size 128.
- The quantization format is NormalFloat. It uses a tensor-level table T = [v_0 … v_{2^b-1}] and a per-group scale s_g, so the group-level table is s_g·T. This "incur[s] almost the same memory overhead as uniform quantization" [V]. The kernel also accepts arbitrary tables: int4/3/2, fp4/3/2, nf4/3/2, or "arbitrary lookup tables" (README) [V].

### 1.2 Offline matrix restructuring (the memory-layout story)
- Tensor-core MMA "require[s] the input matrices to adhere to specific layout specifications within the registers of 32 threads." The dequantized fragment must already be in that layout [V].
- Runtime reordering "introduces a substantial number of operations." FLUTE instead uses the fact that Q is static at inference and reorders the weights **offline**, "such that after dequantization, the weights are already laid out exactly in the expected format." The paper credits prior work: Marlin (Frantar et al. 2024), Xia et al. 2024b (FP6-LLM / Quant-LLM), and Lin et al. 2024 [V].
- **The 3-bit bit-slicing** [V]:
  - The global→shared copies are vectorized: "each thread should access the quantized weight in granularity of 128 bits (or at least in powers of 2)."
  - 3-bit fields don't pack evenly into 128-bit words. Padding them "would be inefficient."
  - So FLUTE splits each 3-bit weight into two bit-slice partitions, "one containing the 1-bit portion and the other the 2-bit portion." It issues "two separate vectorized (asynchronous) data copy instructions." In registers, the two slices are recombined into 3-bit indices before dequantization.
  - The idea is credited to Xia et al. 2024b.
- The paper's own summary of the benefit [V]: restructuring minimizes "bit manipulations associated with unpacking."
- Lecture line [I]: the format (3 bits) is stored as two power-of-two streams (1+2 bits) purely so the memory system can move it with aligned 128-bit transactions. The layout on disk is chosen by the load instruction and the MMA register layout, not by the math.

### 1.3 Vectorized lookup in shared memory
- The table lives in shared memory. Algorithm 1 copies the vectorized table from global to shared memory at the start of each thread block [V].
- The problem [V]: each thread indexes the table with its own codes. This "non-uniform access" puts heavy traffic on shared memory.
- The fix [V]: "we 'vectorize' the lookup operation by accessing two elements at a time." FLUTE builds "an alternative lookup table for every possible pair of values, with each element containing a tuple of 16-bit values." One 32-bit read then returns two dequantized FP16 values, followed by "efficient vectorized scaling operations" (Fig. 2).
- Table sizes:
  - 4-bit [V]: the scalar table has 2^4 = 16 FP16 entries. The pair table has 2^8 = 256 entries of 32 bits, "slightly more than 1KB of storage". That is "a fraction of the 48KB-163KB of shared memory on modern GPUs."
  - 3-bit [I]: 8 entries become 64 pairs × 4 B = 256 B.
- The trade [I]: memory for the table grows from 16 to 256 entries (16× the entries, 32× the bytes). In exchange, the kernel issues half as many shared-memory lookup instructions per weight.

### 1.4 Duplicating the table to reduce bank conflicts
- Background [V]: shared memory has 32 banks, each 32 bits wide, and bank = ⌊addr/32⌋ mod 32 in the paper's notation. Two threads that hit the same bank at different addresses get serialized.
- For the vectorized tables, "a simple implementation could thus cause up to 8-way bank conflicts (4-bit) or 2-way bank conflicts (3-bit)" [V].
- My arithmetic for why [I]: the 4-bit pair table is 256 words over 32 banks, so 8 words share each bank and up to 8 threads can collide. The 3-bit pair table is 64 words, so 2 words per bank.
- Fix [V]: "we duplicate the 4-bit vectorized lookup table multiple times, placing copies in different memory banks, which allows threads to access values from different banks with reduced bank conflicts."
- The number of copies is a tuned parameter, chosen per matrix shape along with tile sizes and pipeline stages [V]. The paper gives no single fixed count.

### 1.5 Stream-K work decomposition
- Standard data-parallel tiling gives one output tile to each thread block, and a block occupies one SM. **Wave quantization** "happens when the number of output tiles is not an even multiple of the number of processor cores," so "the last wave uses only a subset of the cores, leaving the rest idle" [V].
- This is "especially problematic in low-bit and low-batch scenarios, which result in smaller input matrices (activations and quantized weights), thus making the effect of wave quantization more pronounced" [V].
- The Fig. 2 caption [V] says that when "the weight matrix is heavily quantized, the reduced size can lead to 'stragglers' in Slice-K due to uneven workload assignment."
- Stream-K (Osama et al. 2023) "distributes the tiles such that each SM's computations can span beyond specific rows or columns." Multiple thread blocks can "collaboratively compute a single output tile." Fig. 2 example: 35 M-N-K tiles spread over 3 SMs [V].
- **Accumulation detail** [V]: SMs that share an output tile reconcile partial sums through a global scratch space. Accumulation is FP32 in registers, but the global reduction of partial sums is done in FP16, because "writing to global memory in FP32 results in significant traffic."
- Lecture line [I]: in decode, the matrix shrinks by roughly 4× through quantization and also has a skinny batch dimension. There is too little work per tile to fill every SM evenly, so how the work is split matters as much as the bytes saved.

### 1.6 Results

**Kernel benchmarks** [V]
- GPUs: A6000 and A100 (A100-80GB is named in the §2 roofline example).
- Shapes come from Llama-3-8B and Llama-3-70B. Each timing averages 3 random data sets × 100 runs.
- Baselines: LUT kernels (bitsandbytes, BitBLAS-NF4). For reference, uniform-only kernels (Marlin, BitBLAS INT4) are shown as dashed lines.
- Headline (abstract): "At batch sizes < 32 and quantization group size of 128 ..., the FLUTE kernel can be 2-4× faster than existing GEMM kernels."
- Fig. 3 (W4G128): FLUTE is "occasionally nearing the peak theoretical speedup (of 4x) on A6000" over 16-bit torch.mm. "Other LUT-compatible kernels achieve similar speedups only with a batch size of 1, and their performance quickly degrades." FLUTE "also compares favorably to Marlin."
- Fig. 4: consistent speedups across bit widths (including 3-bit) and group sizes at N=K=8192. The alternatives don't support 3-bit.
- The exact per-shape speedup values are in figures, not text. I did not extract numbers from the plots.
- bitsandbytes still allocated memory during some calls; the authors estimate about 2.5% overhead from this (footnote) [V].

**End-to-end, GPT-Fast, batch size 1** (Table 1) [V]
- Setup: prompt length 1, so only decode is measured. torch.compile is on. CUDA Graphs are off because they are "incompatible with FLUTE."
- LLaMA-3 8B, 16-bit baseline: 44.8 tok/s on 1×A6000, 90.2 tok/s on 1×A100.

LLaMA-3 8B, learned NF:

| Config | Bits/param | Size (GB) | WikiText-2 PPL | 1×A6000 tok/s | 1×A100 tok/s |
|---|---|---|---|---|---|
| 16-bit | 16.00 | 15.1 | 6.1 | 44.8 | 90.2 |
| W4G32 | 4.50 | 5.7 | 6.1 | 91.3 (2.0×) | 113.7 (1.3×) |
| W4G64 | 4.25 | 5.5 | 6.1 | 95.9 (2.1×) | 119.4 (1.3×) |
| W4G128 | 4.13 | 5.4 | 6.2 | 98.1 (2.2×) | 121.6 (1.3×) |
| W4G256 | 4.06 | 5.4 | 6.3 | 99.8 (2.2×) | 121.7 (1.3×) |
| W3G64 | 3.25 | 4.7 | 7.2 | 104.1 (2.3×) | 128.5 (1.4×) |
| W3G128 | 3.13 | 4.6 | 7.5 | 108.1 (2.4×) | 133.5 (1.5×) |
| W3G256 | 3.06 | 4.6 | 7.9 | 110.0 (2.5×) | 135.5 (1.5×) |

- W3G32 is 3.50 bits/param, 91.9 tok/s (2.1×) on A6000 and 117.7 tok/s (1.3×) on A100.
- LLaMA-3 70B:
  - 16-bit: 17.2 tok/s on 4×A6000 with tensor parallelism, 19.9 tok/s on 2×A100.
  - W4G128: about 33.1 tok/s on 4×A6000 (1.9×) and 30.3 tok/s on 2×A100 (1.5×). A single GPU gives 14.7 tok/s (A6000) and 18.6 tok/s (A100).
  - W3G128: 32.7 tok/s on 4×A6000 (1.9×), 34.5 tok/s on 2×A100 (1.7×).
- The abstract's summary: "end-to-end throughput increase of 1.5 to 2 times."
- **Lecture point [I]:** the same 4-bit model gains about 2.1–2.2× on A6000 but only about 1.3× on A100. The A100 has much more DRAM bandwidth and a faster FP16 baseline (90 vs 45 tok/s), so fixed overheads (dequantization, lookups, non-GEMM ops) are a larger share of the time. The byte reduction does not turn 1:1 into speed. The authors' own limitation note fits this: on A100, FLUTE "still falls short of the peak performance that kernels specialized for uniformly quantized matrices can achieve."
- Also [V]: a vLLM single-batch latency benchmark (Fig. 5; LLaMA-3 8B/70B and Gemma-2 9B/27B at W4G64 and W3G64) and LLaMA-3.1 405B at W4G64 on a single node (Fig. 6). Those numbers are in figures; I did not extract them.
- An earlier WebFetch summary quoted "~98 tok/s on 1×A6000" as a vLLM figure. That is actually the GPT-Fast Table 1 number for W4G128. **Do not attribute it to vLLM.**

### 1.7 Learned NormalFloat ("NF (learned)", called NFL in the README)
- Standard NF [V]:
  - It takes 2^{b-1} evenly spaced probabilities in [δ, ½] and 2^{b-1}+1 in [½, 1-δ], with δ = ½(1/30 + 1/32).
  - These map to Gaussian quantiles q_i = Φ^{-1}(p_i), normalized to [-1, 1].
  - Each group is absmax-scaled and every weight is rounded to the nearest quantile.
- Learned extension [V]:
  - The table is quantiles of N(0, σ²) with σ = 1/Φ^{-1}(1-δ), so quantization is rewritten as c_j = argmin_i |s·σ̃·q_i − u_j|.
  - σ̃ is initialized to that σ and learned by gradient descent on the negative log-likelihood of calibration data, with a straight-through estimator.
  - After training, s·σ̃/σ is saved as the new scale, so **storage and the kernel stay unchanged**.
  - Calibration: 128 sequences × 2048 tokens from WikiText-2 train.
  - A K-means-style update of the table values gave "no meaningful improvements."
- Quality, Table 2, W4/W3 at group 128 [V]:
  - LLaMA-3 8B WikiText-2 perplexity: unquantized 6.1, NF 6.6 / 9.2, NF (learned) 6.2 / 7.5, AWQ 6.6 / 8.2, GPTQ 6.5 / 9.6.
  - LLaMA-3 70B: unquantized 2.9, NF (learned) 3.1 / 5.2, NF+AWQ 3.2 / 4.6.
  - The authors stress that "the quantization method itself is not the main contribution."

### 1.8 Stated limitations [V]
1. "Mostly optimized for Ampere-generation GPUs." It doesn't use Hopper features, though "the majority of the methods ... could still be applicable." The README lists H100 as "unoptimized" [V].
2. Ampere MMA fragments are [16,16]×[16,8]. At **batch below 16, inputs are padded in shared memory**. This wastes on-chip movement and compute but not DRAM traffic, so memory-bound cases still speed up. "switching to SIMT cores could further enhance performance."
3. Built for memory-bound decode; "performance tends to degrade with larger batch sizes."
4. On A100 it "still falls short of the peak performance that kernels specialized for uniformly quantized matrices can achieve."
5. The discussion section adds a hardware wish list: tensor cores lack **mixed-type input operands** (e.g. FP16 × INT4), and "the lack of in-register dynamic indexing" forces software workarounds. "Enhanced hardware acceleration for indexing into small lookup tables could also prove beneficial."

README practicalities [V]:
- The kernel is **shape-specialized**: tile sizes etc. are tuned per (GPU, shape, dtype, bits, group size). New models need offline tuning that can take "tens of minutes to hours." Experimental auto-tune arrived Jan 2025.
- There are numerical-instability or correctness cases at W4G256 on A100 and on RTX 4090 with bf16.
- bf16 is slower than fp16 on Ampere.
- Supported GPUs: A100, A6000, RTX 4090 (added Aug 2024).
- vLLM integration exists. The HuggingFace path is "mostly experimental and not optimized."
- Nov 2024: vector (de)quantization with vector_size=2 was added for HIGGS (Malinovskii, Panferov, Ilin, Guo, Richtárik, Alistarh, arXiv:2411.17525, NAACL 2025 per the README). HIGGS is data-free Hadamard rotation plus MSE-optimal grids, and it uses FLUTE as its kernel path [V README; abstract A].

---

## 2. LUT-GEMM (LUT over activation partial sums, on GPU)

**Citation:** Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, Dongsoo Lee (NAVER Cloud / POSTECH). "LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models." arXiv:2206.09557 (v4, Apr 2024). ICLR 2024. Code: github.com/naver-aics/lut-gemm.
The ICLR venue is widely cited and the v4 timing fits, but I did not see it stated in the HTML text [I, high confidence]. The author list is from the HTML author block; order partly inferred.

- **Format: binary-coding quantization (BCQ)** [V]. A q-bit weight vector is w ≈ Σ_{i=1..q} α_i b_i, with α_i ∈ ℝ+ and b_i ∈ {−1,+1}^n. The paper extends BCQ with a bias term z, which lets it represent uniform (INT) quantization too: "we are the first to show that prior uniform quantization can be reformulated in the form of BCQ."
- **Mechanism** [V]:
  - Because B is binary, rows of B·x repeat the same partial sums.
  - For each sub-vector of μ activations, precompute all 2^μ signed sums into a table. μ=3 gives 8 values; the GPU example in Fig. 2 uses μ=8.
  - A key is formed by concatenating μ bits of B, and the partial dot product is fetched from the table.
  - Partial products are summed and then multiplied by the scaling factors α.
  - Complexity is O(m·n·q/μ), compared with O(mn), for "a computational savings of q/μ times" when mq ≫ 2^μ. With q bit-planes the cost is linear in bit width.
  - Tables live in shared memory. The appendix quotes "19TB/s for A100" shared-memory bandwidth and "only 1KB is required for every 8 hidden dimensions" [V].
- **Claimed advantage** [V]: it "eliminates the resource-intensive dequantization process" and "reduces computational costs."
- **Numbers** [V]:
  - Table 1: OPT-175B first FFN layer (4m×m, m=12288, g=128), single batch, A100-80GB.

    | Kernel | Latency | Speedup |
    |---|---|---|
    | cuBLAS FP16 | 0.7256 ms | 1.00× |
    | cuBLAS INT8 | 0.6345 ms | 1.14× |
    | OPTQ INT3 (dequant+GEMM) | 0.3599 ms | 2.02× |
    | AWQ INT4 | 0.3238 ms | 2.24× |
    | LUT-GEMM INT4 | 0.2688 ms | 2.70× |
    | LUT-GEMM INT3 | 0.2250 ms | 3.22× |

  - Table 4, end-to-end per-token latency for OPT-175B in FasterTransformer, A100 80GB:
    - FP16 needs 8 GPUs: 42.4 ms.
    - 3-bit on 1 GPU: OPTQ 106.5 ms vs LUT-GEMM 51.6 ms. That is the headline "2.1×".
    - 1 GPU, LUT-GEMM at 1/2/3-bit: 30.4 / 40.1 / 51.6 ms.
  - Main pitch: take OPT-175B from 8 GPUs to 1 at comparable latency.
- **Limitation, stated** [V]: "primarily focuses on single-batch inference and exhibits diminishing performance gains as the batch size increases... attributed to the constrained memory bandwidth between core and LUTs in the shared memory."
- **Independent verdicts on GPU** (useful counterpoint):
  - The T-MAC paper [V]: on Llama-2 shapes on A100, LUT-GEMM's average latency is **2.34×, 1.87×, 1.75× longer** than BitBLAS dequantization kernels for W4A16, W2A16 and W1A16 mpGEMV. It attributes this to the GPU "offer[ing] either inadequate storage capacity for the lookup tables or insufficiently rapid table access."
  - The LUT Tensor Core paper [V], Fig. 4: LUT-GEMM underperforms the CUTLASS dequantization kernel on A100 on LLaMA2-70B shapes.
  - LUT-GEMM's own comparisons were against the older OPTQ and AWQ kernels.
- **How it differs from FLUTE** [V + I]:
  - LUT-GEMM's table is built **online from activations** and replaces the multiply-adds. The tensor cores are bypassed; it runs on CUDA cores and shared memory.
  - FLUTE's table is **fixed, built from weight values** and replaces only the dequantization. The multiplies still run on tensor cores.
  - LUT-GEMM needs a BCQ-representable format (binary planes). FLUTE accepts any scalar codebook of 2^b values.
  - LUT-GEMM's cost scales with bit-planes. FLUTE's cost is roughly flat in bits apart from bytes moved.

---

## 3. T-MAC (bit-serial LUT mpGEMM on CPUs)

**Citation:** Jianyu Wei, Shijie Cao, Ting Cao, Lingxiao Ma, Lei Wang, Yanyong Zhang, Mao Yang (USTC / Microsoft Research). "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge." EuroSys '25. arXiv:2407.00088. Code: github.com/microsoft/T-MAC [V]. The author list past the first three is from memory [I].

- **Key idea** [V]:
  - Turn "data-type-centric multiplication" into bit-wise table lookup: A×W = Σ_i 2^i · A×W_i, where each W_i is a one-bit matrix.
  - For each group of g activations, precompute all 2^g signed sums. With g=4 that is a 16-entry table, from −A1−A2−A3−A4 to +A1+A2+A3+A4.
  - Each g-bit group of a weight bit-plane indexes that table.
  - Result: "no multiplication," fewer additions, and no dequantization. "Our LUT-based kernels scale linearly to the weight bit-width."
  - One layout covers every bit width, whereas dequant kernels need a custom layout per width (the paper notes W3 layouts pack "2 bits and the other 1 bit in separate").
- **Hardware fit** [V]:
  - Tables are kept **in registers**, and the lookup uses byte-shuffle instructions: ARM NEON `TBL` (`vqtbl1q_u8`) and x86 AVX2 `PSHUF`.
  - "LUT for g=4 exactly fits into one register for ARM.TBL/AVX2.PSHUF": a 128-bit register holds 16 int8 entries.
  - AVX2's 256-bit shuffle works in two 128-bit lanes, so the table is **duplicated** to fill both lanes and 32 indices are looked up per instruction. This parallels FLUTE's table duplication [I].
- **Supporting tricks** [V]:
  - A LUT-centric data layout (axis reordering, tiling, and weight tiles permuted to sit contiguously in memory) keeps tables resident without register spills.
  - **Table quantization**: FP16 table entries become int8 with a scale, using fine-grained dynamic quantization.
  - **Mirror consolidation**: the table is symmetric, so only half is stored. Combined, "up to a quarter of its original size."
  - An ablation on M2-Ultra: without memory optimizations, the intrinsic-based lookup alone ("TM-base") was up to 17% *slower* than llama.cpp. Table quantization, tiling, permutation and related steps made it win. **Layout, not the idea alone, is what makes it fast.**
- **Numbers** [V]:
  - Kernel speedup vs llama.cpp is "up to 6.6× and an average of 3.6×". The abstract says up to 4× throughput and 70% less energy; the intro says 60–70%.
  - End-to-end: 2.8× for Llama-2-7B 2-bit.
  - BitNet-b1.58-3B: 30 tok/s on 1 core and 71 tok/s on 8 cores of an M2-Ultra; 11 tok/s on a Raspberry Pi 5 (11.1 in the intro).
  - Multithreaded mpGEMM at sequence length 256, 2-bit: up to 4.0×, 5.3× and 5.3× on RPi, Orin and Surface. M2-Ultra is the exception because of Apple's AMX coprocessor; there T-MAC reaches 2.0× at 1-bit.
  - On Jetson AGX Orin, T-MAC on CPU "significantly outperforms" llama.cpp on GPU for W1A16 and is comparable at W2/W3. The GPU wins at higher bit widths and larger shapes.
  - Devices tested: M2 Ultra, Jetson AGX Orin, Surface Book 3, Raspberry Pi 5.
- **Decode-cost angle** [I]: on CPUs the dequantize-then-dot path is compute-bound enough that the paper reports (Fig. 6) that dropping from 4 to 1 bit "even increases latency cost for most of the cases" for dequant kernels. T-MAC makes cost fall with bits. This is the CPU-side mirror of the lecture thesis.

---

## 4. LUT Tensor Core (hardware/software co-design)

**Citation:** Zhiwen Mo, Lei Wang, Jianyu Wei, Zhichen Zeng, Shijie Cao, Lingxiao Ma, Naifeng Jing, et al. (Microsoft Research and others). "LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference." ISCA '25 (Tokyo). arXiv:2408.06003 (v3, Jul 2025). Code: github.com/microsoft/T-MAC/tree/LUTTensorCore_ISCA25 [V].

- **Diagnosis of GPU software LUTs** [V]. There are two problems:
  1. The best lookup instruction, `prmt` (permute), "has a limited width that prevents completing a whole table lookup in a single instruction."
  2. Where to put the table: registers force duplication across threads (register spills), while shared memory "may result in bank conflicts due to random accesses by threads within a warp."

  Hence LUT-GEMM loses to CUTLASS dequantization kernels on A100 (Fig. 4).
- **Co-design** [V]:
  - *Software:* table precomputation becomes a separate operator, fused with the previous op, instead of being redundantly recomputed per unit. {0,1} is reinterpreted as {−1,+1} so symmetry halves the table. Table quantization handles width and activation bit widths.
  - *Hardware:* a LUT-based tensor core with a bit-serial circuit for arbitrary weight bit widths, and an **elongated tile shape**. N should be large (e.g. 64/128) so each table entry is reused by many MUXes; K should stay small (e.g. 4) because table size grows as 2^K.
  - *ISA/compiler:* a new **LMMA** instruction extending MMA with operand-type and shape metadata, compiled through TVM, Roller and Welder.
- **Numbers** [V]:
  - The LUT-based tensor core takes 4–6× less power and area than a conventional tensor core. In Accel-Sim it "occupies only 16% of the area of a conventional Tensor Core while achieving even higher mpGEMM performance."
  - Versus LUT-GEMM software: up to 1.42× on GEMV and 72.2× on GEMM.
  - Versus the prior LUT accelerator (Lee et al. 2019, UNPU): 1.44× compute density and energy efficiency.
  - End-to-end, via their own tile-level simulator: up to 8.2× over a W_FP16A_FP16 tensor core. Even the 8× configuration uses 38.3% of the area.
  - A W_INT1A_FP16 LUT tensor core is 58.4% of the area of the FP16 one, with 4× the theoretical FLOPs.
  - All of this is **simulation** (Accel-Sim plus a custom simulator), not silicon.
- **Lecture use** [I]: this answers FLUTE's closing wish ("native support for such instructions in future hardware could be beneficial"). If the format is LUT-shaped, the ideal endpoint is a tensor core that takes the compressed operand natively.

---

## 5. Other related work: the decode-cost angle

**QTIP** — Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa. "QTIP: Quantization with Trellises and Incoherence Processing." NeurIPS 2024. arXiv:2406.11235.
- It designs the code *for the decoder* [V]:
  - Vanilla trellis-coded quantization needs the graph and a 2^L×V codebook ("too large to fit in cache") and decodes sequentially.
  - QTIP uses a "bitshift trellis": the next state comes from a shift by kV bits, which allows parallel decoding and needs no stored graph.
  - It adds *computed* pseudorandom Gaussian codes:
    - "1MAD" (an LCG plus a sum of bytes)
    - "3INST" (an LCG plus an XOR/mask to reinterpret bits as FP16)
    - "HYB" (a hash into a small 2D LUT)
  - Cost is "≤4 instructions per weight"; QTIP's codes use "as few as 2 instructions per weight."
- HYB uses Q=9, a **2 KiB codebook that "fits in L1 cache even after duplication for bank conflicts (32×)"** [V]. It uses the same trick as FLUTE, and its 2D codebook is chosen because "shared memory ... is accessed in 32-bit-word elements, and each such word can contain two 16-bit floats." This is the same reasoning as FLUTE's paired table [V].
- Decode reaches "over 80% of peak memory bandwidth" [V].
- Table 4, batch-1 decode on RTX 6000 Ada (960 GB/s) [V]:

  | Model | FP16 | AQLM 2-bit | QuIP# 2-bit | QTIP 2/3/4-bit |
  |---|---|---|---|---|
  | Llama-2-7B | 55.9 tok/s | 81.5 | 186 | 188 / 161 / 140 |
  | Llama-2-70B | OOM | 8.78 | 22.2 | 23.5 / 19.1 / 16.3 |

- QTIP matches QuIP# throughput with a 256-dimensional effective code, 32× the dimension, "with no additional inference-time cost."

**AQLM** — Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh. "Extreme Compression of Large Language Models via Additive Quantization." ICML 2024. arXiv:2401.06118.
- Weights are sums of M codewords from learned codebooks over groups of 8.
- The 1×16 configuration (a 2^16-entry codebook of 8-dim FP16 vectors) is 1 MiB, which "does not fit in L1 cache" (per QTIP) [V].
- AQLM's own numbers show the decode tax [V].
  - Table 14, RTX 3090, batch 1, Llama-2-7B:

    | Config | tok/s |
    |---|---|
    | FP16 | 54.2 |
    | 1×16 (about 2.3 bits, roughly 7× smaller) | 65.3 (**only about 1.2×**) |
    | 2×8 (smaller codebooks that fit in cache, slightly worse accuracy) | 114.1 |

  - The 70B model: 5.8, 6.7 and 14.3 tok/s for FP16, 1×16 and 2×8.
  - Layer-level (Table 5): 1×16 gives ×1.20–1.31 over FP16. The authors write: "multiple smaller codebooks allow efficient GPU cache utilization, leading to greater speedup, at the price of slightly lower accuracy."
  - On CPU (i9, 8 cores), 2×8 / 4×8 / 8×8 codebooks give up to about 4× over FP32.
- **The clearest single data point for the lecture** [I]: a roughly 7× compression (about 2.3 bits for 7B) that buys about 1.2× speed, because the decoder's table doesn't fit where it's needed.

**SqueezeLLM** — Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, Kurt Keutzer. "SqueezeLLM: Dense-and-Sparse Quantization." ICML 2024. arXiv:2306.07629. Author list beyond the first two is from memory [I].
- Sensitivity-weighted k-means gives a non-uniform **per-output-channel LUT** at 3/4 bits. A small fraction of outliers and sensitive weights is kept in FP16 in CSR format [V].
- The kernels [V]:
  - A 3/4-bit CUDA LUT **matrix-vector** kernel (indices → FP16 table entries, then FP16 arithmetic).
  - A balanced sparse CSR kernel with about 10 nonzeros per thread, because nonzeros are skewed across rows.
  - Both launch in one call.
- Numbers on A6000, generating 128 tokens [V]: up to 2.4× over FP16 for dense-only (the abstract says "up to 2.3×"). Keeping 0.45% of weights sparse adds about 10% latency, still up to 2.2× over FP16.
- Layout point [V]: GPTQ with activation reordering "incurs a significant latency penalty as elements in the same channel are associated with different scaling factors, resulting in distributed memory accesses." This is a direct example of a format choice that costs decode speed.
- The limit [I, consistent with FLUTE's framing]: these are GEMV kernels for batch 1, with no tensor-core path. FLUTE explicitly targets extending LUT dequantization to batch up to about 32 with tensor cores.

**bitsandbytes NF4** (Dettmers et al., QLoRA, NeurIPS 2023, arXiv:2305.14314)
- NF4 is a 16-entry table of normal quantiles, with blockwise absmax scaling (docs [V]).
- In bitsandbytes 0.45.0 source (`autograd/_functions.py::matmul_4bit`) [V]:
  - When the input is a single token (`A.numel() == A.shape[-1]`) and no gradient is needed, it calls a fused `gemv_4bit` kernel.
  - Otherwise, including any batch > 1, it goes through `MatMul4Bit`, which **dequantizes the whole weight to FP16 and calls a normal matmul**.
  - So the byte savings vanish as soon as batch > 1.
- Current `main` (2026) routes inference through a newer fused `torch.ops.bitsandbytes.gemm_4bit` op [V source]. I have not checked its performance.
- FLUTE's benchmark [V]: bitsandbytes reaches FLUTE-like speedups "only with a batch size of 1, and their performance quickly degrades."

**Marlin** (Frantar, Castro, Chen, Hoefler, Alistarh, arXiv:2408.11743), the uniform-INT4 reference point
- Per FLUTE [V]: "up to 4×" matmul speedups "even in moderate batch (16-32) settings." It relies on "highly tuned PTX assembly instructions" for INT→FP conversion, which "are not applicable to LUT-based dequantization."
- FLUTE borrows Marlin's offline-reordering idea.
- I did not read the Marlin paper itself for this slice.

---

## 6. Synthesis for the lecture [I, built on the [V] items above]

1. **The code format sets the decoder's instruction budget.**
   - Uniform INT4 → FP16 can be done with a few bit-trick PTX instructions (Marlin).
   - A 16-entry table needs a *dynamic index*, which registers can't do, so the table goes to shared memory. That brings shared-memory bandwidth and bank conflicts into the picture (FLUTE, LUT Tensor Core).
   - A 2^16-entry, 1 MiB codebook doesn't fit in L1 at all (AQLM 1×16 → about 1.2× speed despite about 7× smaller).
   - QTIP goes the other way: it designs the code so that decoding is 2–4 ALU instructions and uses a 2 KiB table.
2. **Kernel authors use the same tricks independently.**
   - *Pair the table entries* to fill 32-bit shared-memory words (FLUTE's 256-entry pair table; QTIP's V=2 HYB code).
   - *Duplicate the table* to dodge bank or lane limits (FLUTE's copies across banks; QTIP's 32× duplication; T-MAC's 128→256-bit lane duplication).
   - *Reorder offline* so that loads are aligned and land in the MMA register layout (FLUTE, Marlin, T-MAC's permuted tiles).
   - *Split odd bit widths into power-of-two bit-planes* (FLUTE's 1+2 split for 3-bit; T-MAC's and LUT-GEMM's bit-serial planes).
3. **Where the table sits depends on the hardware.**
   - GPUs have tensor cores but weak dynamic indexing, so the best GPU LUT kernels (FLUTE, QTIP HYB) use the LUT only to *dequantize* and keep FP16 tensor-core math.
   - CPUs have cheap in-register byte shuffles (TBL/PSHUF) and weak matmul, so T-MAC uses the LUT to *replace the math*.
   - Activation-partial-sum LUTs on GPUs (LUT-GEMM) lose to dequantization kernels on A100 according to two later papers.
   - The logical endpoint is new silicon (LUT Tensor Core, simulated).
4. **Small-batch decode also has a scheduling problem, not only a bandwidth one.** Quantized weights plus a skinny M leave too few tiles to fill every SM, hence Stream-K (FLUTE). Below batch 16, the MMA shape forces padding in shared memory.
5. **Speedup depends on the GPU, not only on bits.** The same FLUTE W4G128 Llama-3-8B gives 2.2× on A6000 but 1.3× on A100 (Table 1).

## Things I could not verify or did not find
- The exact number of table copies FLUTE uses. It is tuned per configuration and not given as a number.
- FLUTE per-shape speedup values vs bitsandbytes, BitBLAS and Marlin (figures only, not extracted).
- FLUTE vLLM figure values (Fig. 5 and 6, not extracted).
- Marlin's own reported numbers (paper not read here).
- Per-GPU numbers for the current bitsandbytes `gemm_4bit` path.
