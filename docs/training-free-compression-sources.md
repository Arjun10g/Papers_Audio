# Sources: Compression Without Retraining

Literature-review notes behind `lectures/training-free-compression.md`, written by four Sonnet review agents on 2026-10-02. Tags: [V] verified in full text, [A] abstract or snippet only, [S] secondary, [I] the agent's own inference or arithmetic. Spot-check before citing.


---

# Training-free PTQ: state of the art as of Oct 2026 (lit review notes)

Tags: [V] verified in full text of the cited paper (I downloaded and read the PDF text); [A] abstract only; [S] secondary/search snippet; [I] my inference or background knowledge not re-checked here.
Raw texts are in ./q/ (arXiv ids below). All numbers are copied from the cited tables. "FT" = fine-tuning.

## 0. Framing and terminology (read first)
- "Training-free" has three tiers in this literature. (a) Data-free / RTN-style: no calibration data (QuaRot+RTN, HIGGS, NVFP4 RTN). (b) Calibration-only, forward pass / closed-form: GPTQ, AWQ, DuQuant, GPTAQ, KVQuant, KIVI-style. (c) Calibration + gradient optimisation of auxiliary parameters (rotations, affine transforms, rounding offsets) while the weights themselves are not trained: SpinQuant, FlatQuant, OSTQuant, SignRound, OmniQuant. Tier (c) is not "training-free" in the strict sense; I label it "calibrated/learned" below. [I]
- GPTAQ's own definition: "finetuning-free ... circumvents gradient-based optimization in favor of forward-pass calibration computation" [V, 2504.02692 Sec.1].
- No single leaderboard exists. The closest are PTQ-Bench (2502.13178), "Quantization Hurts Reasoning?" (2504.04823), "Give Me BF16 or Give Me Death" (2411.02355) and the MR-GPTQ study (2509.23202). Methods are mostly compared in each paper's own tables, often with copied baseline numbers; treat "SOTA" claims as paper-relative. [I]

## 1. Baseline short note: SliceGPT
See slicegpt_core.md (kept unchanged). One line for this lecture: SliceGPT is structured pruning; its 25% slicing of Llama-2 70B costs 3.32 -> 4.60 WikiText-2 PPL [V, 2401.15024 Table 1], which is much worse in quality-per-bit than 4-bit quantization (Llama-2 70B W4 weight-only costs ~0.1-0.2 PPL, see KronQ table below).

## 2. Weight-only PTQ

### GPTQ (2210.17323) and successors
- Idea: layer-wise Hessian-based error compensation; quantize columns sequentially and update remaining weights. [I] (GPTQ itself not re-read; described in GPTAQ/KronQ as "layer-wise independent calibration" [V]).
- Cost: GPTQ part is cheap: 0.2 GPU-h Llama-3-8B, 1.8 GPU-h Llama-3-70B on one A100 [V, GPTAQ Table 2].
- **GPTAQ** (Li et al., ICML 2025 [S venue], 2504.02692): asymmetric calibration, matches quantized layer output to the full-precision model's output so errors from earlier layers are corrected; closed form, ~20 more lines than GPTQ, "finetuning-free". Cost 0.3 vs 0.2 GPU-h (8B), 2.7 vs 1.8 (70B). 3-bit g-wise: Llama-2-7B Wiki2 6.59 vs GPTQ 6.88 vs AWQ 6.75 (FP16 5.47) [V, Table 3]. Quantized Llama-3.1-405B to W4A4 on a single GPU: PPL 3.48 vs GPTQ 5.82 (FP16 1.44) [V, Table 4]. (Note the arXiv/ICML title says GPTAQ; no separate "GPTQv2" found by me [I].)
- **BoA / TurboBoA** (Samsung, ICLR 2026, 2602.04929): attention-aware Hessians, backprop-free; TurboBoA quantizes several out-channels jointly (>3x faster than BoA), corrects errors propagated from earlier layers, adds adaptive grid. Used on top of QuaRot/SpinQuant/OSTQuant rotations. INT2 weight-only on QuaRot-rotated Llama-3-8B: Wiki2 PPL 13.54 vs GPTQ 18.28 vs BoA 15.24 (FP16 6.139); INT3: 7.116 vs GPTQ 7.490 [V, Table 4].
- **KronQ** (COLM 2026, 2607.07964): adds gradient covariance to the Hessian (Kronecker-factored), bidirectional incoherence processing (rotations on input and output sides), sensitivity-based mixed precision; needs one backward pass over calibration set to estimate gradient covariance, 128 WikiText-2 samples x 2048 [V]. Per-channel weight-only WikiText-2 PPL (their Table 1; baselines mostly copied from papers) [V]:
  - W4: L2-7B 5.56, L2-70B 3.40, L3-8B 6.42, L3-70B 3.25 (FP16 5.47 / 3.32 / 6.14 / 2.85). Competing: SpinQuant 5.58/3.43/6.49/3.49, OSTQuant 5.64/3.41/6.53/3.19, GPTQ L3-70B 27.49 (!), GPTAQ L3-70B 399.46.
  - W3: KronQ L2-7B 5.83, L3-8B 7.09, L3-70B 4.41 vs QuIP# L2-7B 6.19, GPTQ L3-8B 8.24, GPTQ L3-70B 2.6e3.
  - W2: KronQ L2-7B 8.19, L2-70B 5.14, L3-8B 11.92, L3-70B 7.93 vs QuIP# L2-7B 12.30, L2-70B 4.87 (QuIP# still wins on 70B Llama-2), GPTQ L3-8B 25.43, GPTAQ L3-70B NaN.
  Caveat: single-group self-reported results, very new; not independently replicated [I].
- Lesson for the lecture: plain GPTQ per-channel can blow up on Llama-3 (L3-70B W4 27.49 PPL in KronQ's table, W3 2.6e3) without rotations/incoherence processing or grouping [V, KronQ Table 1].

### AWQ (2306.00978)
- Idea: protect ~1% salient weight channels, identified from activation statistics, by per-channel scaling (equivalent transformation) before quantization; no backprop, no reconstruction, so less calibration overfitting [V abstract+text]. Calibration set tiny and insensitive [V, claim "minimal reliance on calibration set"].
- Recommended by the "Quantization Hurts Reasoning?" study as the weight-only method for reasoning models; W4A16 near-lossless (<=1% drop) [V, 2504.04823 Sec.1 findings].
- Weak at low bit: 3-bit g128 Llama-2-7B Wiki2 AWQ 6.75 vs GPTAQ 6.59 [V]; 2-bit collapses (AWQ NaN/1e7 in Qwen3 study and KronQ) [V].
- Qwen3 caveat (2505.02214, per-channel, no groups): Qwen3-8B-Base Wiki2 6.99 FP16 -> AWQ W4 7.73, GPTQ W4 7.63, W3 AWQ 11.4; Qwen3 degrades more than Llama-3 at <=3 bits (AWQ w3g128 C4 PPL 10.4->23.8 for Qwen3-8B vs 9.2->11.6 for Llama-3-8B) [V].

### Vector / trellis quantization: QuIP#, AQLM, QTIP
- QuIP#: Hadamard incoherence + E8 lattice codebook; AQLM: additive multi-codebook learned quantization. Both published results use blockwise fine-tuning [V via QTIP Sec.4; descriptions [I]].
- **QTIP** (NeurIPS 2024, 2406.11235): trellis-coded quantization with incoherence processing; stateful decoder decouples codebook size from bitrate, so effective dimension is very high (~256); computed codes (1MAD, 3INST, hybrid) are decodable fast on GPU. Without fine-tuning, QTIP 3INST beats QuIP#-with-FT and AQLM-with-FT at almost all sizes [V, Table 3]. Llama-2-7B Wiki2 PPL (FP16 5.12): 2-bit no-FT 6.82 (3INST) / 7.05 (1MAD) vs QuIP# no-FT 8.22 vs QuIP# FT 6.19 vs AQLM FT 6.14; 3-bit no-FT 5.38 ; 4-bit 5.17 [V]. Llama-2-70B 2-bit no-FT 3.90 (FP16 3.12) [V]. Llama-3-8B (ctx 8192, FT'd hybrid code assumed [I]) 2-bit Wiki2 7.33 vs QuIP# 7.84, BF16 5.54; 3-bit 6.01; 4-bit 5.67 [V, Table 7]. Llama-3.1-405B-Instruct: 2-bit PPL 3.29 vs BF16-like "FP8" 1.70, 4-bit 1.79 [V, Table 8]. Decoding on 2-7B ~188 tok/s at 2 bits vs 55.9 FP16 [V, Table 4].
- Quantization cost (from a competitor's table): QuIP# 270 GPU-h (does not fit one A100-80GB), AQLM 336 GPU-h for Llama-2-70B [V, SignRoundV2 Table 5]. QTIP cost not extracted; it inherits QuIP#'s pipeline, so treat as heavy [I].
- Status: QTIP/QuIP#/AQLM remain the quality leaders at 2 bits when expensive procedures and FT are allowed; they are not cheap and not strictly training-free once FT is used.

### HIGGS (data-free; 2411.17525, Yandex/ISTA/KAUST)
- Idea: "linearity theorem" ties layer-wise MSE to perplexity increase; Hadamard rotation makes weights Gaussian, then MSE-optimal (vector) grids; no calibration data; dynamic-programming bit allocation across layers. Beats NF4/HQQ/AF data-free [A+V].
- Llama-3.1-8B Wiki2 (FP16 5.607): 4.02 bits HIGGS(p=2) 6.015, NF 6.225, HQQ 8.057; dynamic data-free 4.00 bits 5.910 (vs GPTQ 4.02 bits 6.238); 3.25 bits dynamic 6.388 vs GPTQ 7.133 [V, Table 3]. Llama-2-7B 2-bit (GPTQ+HIGGS p=2) 8.637 vs QTIP 6.82 [V, Table 2].
- Calibration cost: none. Selling point for the lecture: data-free is within ~0.3 PPL of calibrated GPTQ at 4 bit.

### SignRoundV2 (Intel, 2512.04746) - learned, not training-free [I]
- Signed-gradient tuning of rounding/scales plus a gradient-based DeltaLoss bit allocation. 2.5 h on Llama-2-70B (6 h enhanced) vs EfficientQAT 41, QuIP# 270, AQLM 336 GPU-h [V, Table 5]. W2A16 5-task avg (FP16 Llama-3-70B 75.28): Ours* 70.16 at g128; AQLM 1x16 70.10; QuIP# not reported for L3; GPTQ g128 on Llama-2-70B 34.38 [V, Table 1].

### Other 2026 sub-4-bit entrants (abstract/snippet level only)
- D2Quant (2602.02546), CLAQ, ReQuant etc. claim better sub-4-bit weight-only PTQ [S]. Not verified.

### Who is SOTA, weight-only (my synthesis [I], evidence in tables above)
- 4-bit: essentially solved. GPTQ/AWQ/RTN-g128 are within ~1% of FP16 on average accuracy for 8B+ models ("Give Me BF16": W4A16-INT recovers 99.36% on average across Llama-3.1 8B/70B/405B [V, 2411.02355 Sec.4.1]; Quantization Hurts Reasoning: <=1% drop [V]). Best PPL: rotation+GPTQ-class (SpinQuant/OSTQuant/KronQ ~ +0.1-0.4 PPL on L3-8B; FP16 6.14 -> 6.42-6.53) and QTIP (5.67 vs 5.54 on Llama-3-8B ctx 8192). Data-free HIGGS/NF4 within ~0.3-0.4 PPL.
- 3-bit: QTIP (Llama-3-8B 6.01 vs 5.54 FP) and KronQ (7.09 vs 6.14 FP, different ctx) lead; GPTQ per-channel unstable on Llama-3-70B.
- 2-bit: QTIP / QuIP#/AQLM (with or without FT) for quality (Llama-3-8B 7.33 QTIP ctx 8192); KronQ and SignRoundV2 give much cheaper calibrated alternatives (KronQ L3-8B 11.92, L3-70B 7.93 per-channel); TurboBoA on rotated models 13.54 (L3-8B). Still far from FP16; PTQ-Bench argues "2-bit large model is worse than 4-bit small model" (LLaMA2-70B 2-bit OmniQuant 10.36 PPL/38.46% vs LLaMA2-7B 4-bit 6.55/51.11%) [V, 2502.13178 Sec.4].
- PTQ-Bench finding: compensation-based (GPTQ) is the most uniformly robust strategy; rotation excels at 2 bit on Llama-1/2, GPTQ better on fully-trained Llama-3/3.1 (2-bit LLaMA3-70B GPTQ 23.43 PPL vs QuIP 53.18); QuaRot+GPTQ strong across structures [V, 2502.13178 Secs.3.2, 4].

## 3. Weight + activation PTQ

### SmoothQuant (2211.10438)
- Idea: migrate activation outliers into weights with per-channel scales; W8A8 [I]. Suffices for 8-bit [V, 2504.04823 recommendation]. Collapses at W4A4: Qwen3 PPL 1e4 [V, 2505.02214]. Even W8A8 degrades Qwen3 more than weight-only [V].

### QuaRot (2404.00456; Ashkboos, Hensman et al. - same group as SliceGPT)
- Idea: computational invariance (from SliceGPT) used to insert Hadamard rotations into the residual stream, FFN input to down-proj, and KV cache, spreading outliers so everything is 4-bit; rotations fused into weights, extra online Hadamards for down-proj/attention. Training-free; RTN needs no calibration data, GPTQ variant uses calibration. Llama-2-70B W4A4KV4 <=0.47 Wiki2 PPL loss, 99% zero-shot retained [V, abstract]; lossless at 6/8 bit with RTN, no calibration [V].
- Llama-3-8B W4A4KV4 (FlatQuant's reproduction): RTN Wiki2 10.60 (FP16 6.14), GPTQ 8.16; zero-shot avg 61.34 (RTN)/65.79 (GPTQ) vs 73.23 FP16; Llama-3-70B RTN 55.44 PPL, 35.36% acc (breaks), GPTQ 6.60 [V, FlatQuant Tables 1-2].
- NVFP4/MXFP4 caveat: QuaRot on MXFP4 Llama-3.1-8B-Instruct W4A4 only 79.70% recovery vs MR-GPTQ 93.31% [V, 2509.23202 Table 1].

### SpinQuant (2405.16406; Meta) - learned, not training-free
- Idea: same rotation invariances as QuaRot, but rotations R1/R2 are optimised on the quantized-activation network with Cayley SGD on the Stiefel manifold (weights frozen, then GPTQ/RTN) [V]. 800 WikiText-2 samples x 100 iterations; ~13/18/30 min for Llama-3 1B/3B/8B, ~25/30 min Llama-2 7B/13B, ~3.5 h Llama-2-70B; GPTQ cost extra [V]. Authors: random rotations vary by up to 13 zero-shot points, so learning rotations matters [V]. Robust to calibration set (C4 vs Wiki) [V, App. A.3].
- Verdict on "training-free": no weight training, but gradient-based optimisation of rotation parameters on calibration data. I would call it "calibrated, not training-free in the strict sense"; GPTAQ labels it FT-Free = X (i.e. fine-tuning-based) [V, GPTAQ Table 2 col "FT-Free" shows ✗ for SpinQuant].
- Llama-3-8B W4A4KV4 Wiki2: 7.96 (RTN) / 7.39 (GPTQ); zero-shot avg 66.98 / 68.70 [V, FlatQuant Tables 1-2]; Llama-3-70B 7.58/6.21 PPL, 65.66/71.66 avg [V].

### DuQuant (2406.01721)
- Idea: block-wise rotations built greedily from outlier-dimension positions + zigzag permutation + second rotation to handle "massive outliers"; closed-form, calibration-light, training-free [V abstract]. Cost 2.1 GPU-h Llama-2-7B, 15 GPU-h 70B (GPTAQ's measurement) [V, GPTAQ Table 2]. Weaker on Llama-3 than SpinQuant/FlatQuant: Llama-3-8B W4A4KV4 Wiki2 8.14, avg acc 67.13; MMLU 50.77 vs FP16 62.07 [V, PrefixQuant Tables 2, 4]. Reproducibility note: GPTQ-style GQA models blew up in TurboBoA's runs (DuQuant RTN Llama-3-8B W3: 10.78 PPL; W2 2.6e4) [V, 2602.04929 Table 4].

### FlatQuant (ICML 2025, 2410.09426; Huawei) - learned
- Idea: per-layer learnable affine transforms, Kronecker-factored (two small matrices) to make weights and activations flat; learned with a lightweight objective (AdamW, 15 epochs, 128 WikiText-2 sequences x 2048), fused into one kernel [V]. Calibration time: Llama-3-8B 0.90 h, Llama-3-70B 5.94 h (W+A) on one GPU [V, Table 5].
- W4A4KV4 results (Wiki2 PPL / zero-shot avg-6 tasks), FP16 in brackets: Llama-3-8B RTN 6.98 / 71.23 (6.14 / 73.23); GPTQ 6.90 / 71.33. Llama-3-70B RTN 3.78 / 79.01 (2.86 / 79.95); GPTQ 3.77 / 78.58. Llama-2-70B RTN 3.55 / 76.62 (3.32 / 77.05) [V, Tables 1-2]. Abstract: "less than 1% accuracy drop for W4A4 on LLaMA-3-70B, surpassing SpinQuant by 7.5%" [V]. Speed: up to 2.30x prefill, 1.76x decode vs FP16 at batch 64 on RTX 3090 [V].
- KV cache only on Llama-3-8B: K4V4 6.20 PPL (FP16 6.14); K2V2 8.93 [V, Table 12]; Llama-2-7B K2V2 6.66 vs QuaRot 9.23 [V, Table 13].
- Aggressive: W3A3KV3 Llama-3-8B 10.82 PPL (QuaRot 686) [V, Table 14].

### OSTQuant (2501.13987) - learned
- Learnable orthogonal + scaling transformations with a KL-based objective to improve distribution fit under limited calibration data [V abstract-level]. Used in TurboBoA/KronQ tables (Llama-3-70B W4 weight-only 3.19 PPL vs SpinQuant 3.49) [V, KronQ Table 1]. Detailed W4A4 numbers not extracted.

### PrefixQuant (2410.05265)
- Idea: outlier tokens (e.g. BOS-like, delimiter tokens) carry 94.7% of the quantization error in a Llama-2-7B block; prefix those tokens into the KV cache (training-free, 0.2 min for Llama-3-8B, 1 min for 70B), which also enables static per-tensor activation quantization; optional block-wise fine-tuning (2.2 h on 8B, 17 h on 70B) [V]. W4A4KV4 Llama-3-8B Wiki2 7.26 (O1) / 7.43 (static, O2) vs SpinQuant 7.36, DuQuant 8.14, QuaRot 8.41 (FP16 6.14); zero-shot avg 71.31 vs SpinQuant 68.23 [V, Table 2; unclear whether these use the fine-tuned variant]. Llama-3-70B 4.16 vs QuaRot 6.82 (FP16 2.85) [V].

### Newer 2026 W4A4 entries (abstracts + search snippets only)
- ReSpinQuant (2604.11080): layer-wise rotations merged offline; claims Llama-3-8B W4A4 Wiki2 7.24 vs SpinQuant 7.50, QuaRot 7.82 [S, search snippet]. WUSH (2512.00956, Chen...Hoefler, Alistarh): closed-form data-aware blockwise transform (no gradient descent); +2.8 avg points over Hadamard RTN on Llama-3.1-8B-Instruct MXFP4 and +0.7 over GPTQ-Hadamard, up to 5.8x layer throughput over BF16 [V abstract]. TwinQuant (2606.01556), MosaicQuant (2606.15652), KronQ W4A4 also exist [A].

### Who is SOTA at W4A4 (my synthesis [I])
- Quality: FlatQuant (learned, Llama-3-8B Wiki2 6.90-6.98, avg 71.2; Llama-3-70B 3.78) and PrefixQuant are the strongest widely replicated INTEGER W4A4 results; SpinQuant next; QuaRot/DuQuant weaker on Llama-3. Independent confirmation: the "Quantization Hurts Reasoning?" study picked FlatQuant as the 4-bit leader (in its tables, 4-4-4 (W4A4KV4) on a small DeepSeek-R1-distill model shows a -10.9 average drop for FlatQuant but -46.2 for QuaRot; larger models lose less, e.g. -3.0 vs -5.6 on one 70B-class run) [V, 2504.04823 tables; model identity per row not re-checked].
- Training-free only: QuaRot+GPTQ/GPTAQ (GPTAQ lifts QuaRot Llama-3-70B W4A4 Wiki2 9.44 -> 6.94, avg 62.4 -> 69.1; SpinQuant+GPTAQ 5.00 / 77.7) [V, GPTAQ Table 2], DuQuant, PrefixQuant (training-free variant).
- Honest bottom line: W4A4 integer is NOT lossless: best methods lose ~1-2 points on 70B and ~2 points on Llama-3-8B zero-shot, and more on MMLU and reasoning.

## 4. Hardware formats: FP8, NVFP4, MXFP4
- **FP8 / INT8 (W8A8)**: "Give Me BF16 or Give Me Death" (Red Hat/ISTA, 2411.02355; >500k evals, Llama-3.1 8B/70B/405B): W8A8-FP effectively lossless; W8A8-INT 1-3% loss with tuning; W4A16-INT rivals 8-bit (avg recovery 99.75% 8-bit vs 99.36% W4A16 on the OpenLLM v1 academic suite) [V]. On harder real-world suite 8B W4A16 recovers 96.1%, 70B 97.4% [V, Table 3]. Recommendation: W4A16 cheapest for synchronous/latency use, W8A8 best for high-throughput continuous batching [V abstract]. MR-GPTQ paper: INT8/FP8 RTN recover 99.6-99.7% on Llama-3.1-8B-Instruct W8A8 [V].
- **NVFP4 vs MXFP4** (Egiazarian, Panferov... Alistarh, ICLR 2026, 2509.23202, "Bridging the Gap Between Promise and Performance for Microscaling FP4"): MXFP4 = E8M0 power-of-two scales, group 32; NVFP4 = E4M3 scales, group 16 (my reading plus [S]). Key results [V]:
  - Llama-3.1-8B-Instruct W4A4 avg recovery (MMLU-CoT, GSM8k, HellaSwag, WinoGrande): NVFP4 RTN 94.67%, GPTQ 95.92%, MR-GPTQ 96.08%, SmoothQuant 95.9%; MXFP4 RTN 87.83%, GPTQ 89.47%, QuaRot 79.70%, SpinQuant 88.00%, MR-GPTQ 93.31%; INT4 RTN 92.63%, RTN+Hadamard 94.71%, GPTQ 92.75% (their Table 1). No format is lossless; weight and activation error contribute roughly equally.
  - NVFP4's group-16 "provably neutralizes" Hadamard outlier mitigation: RTN+HT is worse than RTN for NVFP4 (93.82 vs 94.67); the Hadamard helps INT4/MXFP4 (group 32) [V].
  - MR-GPTQ: GPTQ with block-wise Hadamard (group-size blocks), MSE-optimised scales/ format tweaks; kernels (QuTLASS, vLLM) give up to 3.6x layer-wise and 2.2x end-to-end speedup on B200 vs FP16, 6x/4x on RTX5090 [V abstract]. Hypothetical NVINT4 + MR-GPTQ reaches 97.12% (above NVFP4's 96.08%), suggesting INT4-with-microscaling can beat FP4 [V].
  - Real kernels on larger models: Qwen3 models reach >99% recovery with NVFP4 (>=8B); Llama-3.3-70B-Instruct NVFP4 GPTQ 98.13% vs MXFP4 GPTQ+Had 95.23% [V, Sec.4 and Table 6]. W4A16 weight-only Llama-3.1-8B-Instruct: NVFP4 GPTQ 97.96%, AWQ 98.06%, INT4 RTN 97.71% [V, Table 2].
  - Takeaway: "FP4 is not an automatic upgrade over INT4" [V quote from abstract].
- **Four Over Six (4/6)** (2512.02010): NVFP4 tweak: for some blocks scale to max 4 rather than 6 (FP4 grid is non-uniform, big error near max); improves AWQ/SmoothQuant W4A4 NVFP4 perplexity on Llama-3/Qwen3 for all tested models (19.9% and 5.3% closer to BF16), hurts GPTQ in some cases (+34.6% gap increase when combined naively) [V]. Also improves pre-training stability.
- Newer FP4 PTQ papers exist (ScaleSweep 2606.07618, FAAR 2603.22370, MixFP4 2605.31035, FOCUS 2608.01847, a layer-wise NVFP4/MXFP4 sensitivity study 2603.08747) [S: titles only, not read].
- Deployment fact [S, search]: MR-GPTQ is reportedly implemented in NVIDIA TensorRT Model Optimizer as recommended PTQ for MXFP4/NVFP4 [S, emergentmind snippet; unverified].

## 5. KV-cache quantization
- **KIVI** (ICML 2024, 2402.02750): tuning-free asymmetric 2-bit; keys quantized per-channel, values per-token (from outlier-structure analysis); keeps a small recent window in fp16 [I, not re-verified]. Claims 2.6x less peak memory, up to 4x larger batch, 2.35-3.47x throughput on real workloads [V abstract]. Calibration: none.
- **KVQuant** (NeurIPS 2024, 2401.18079): per-channel key quant, pre-RoPE key quant, per-layer sensitivity-weighted non-uniform datatypes (nuqX) derived offline by k-means on a calibration set (16 samples x 2K tokens, WikiText-2), per-vector dense-and-sparse outlier isolation. <0.1 PPL degradation at 3 bits on Wikitext-2/C4; 1M context on one A100-80GB for LLaMA-7B; up to ~1.7x kernel speedup [V abstract]. Training-free, calibration-based.
- **TurboQuant** (Google, 2504.19874): data-oblivious online scheme: random rotation -> Beta-distributed coordinates -> optimal scalar quantiser per coordinate, plus 1-bit QJL residual for unbiased inner products; within ~2.7x of the information-theoretic distortion bound. KV cache: "absolute quality neutrality at 3.5 bits per channel, marginal degradation at 2.5" [V]. LongBench-E avg Llama-3.1-8B-Instruct: full 50.06, TurboQuant 3.5-bit 50.06, 2.5-bit 49.44, KIVI 3-bit 48.50, KIVI 5-bit 50.16; unlike KIVI, also quantizes generated tokens [V, Table 1]. Quantization time ~0.001 s per vector set vs minutes for PQ/RabitQ (d=1536: 0.0013 s vs 239.75 s) [V, Table 2].
- Rotation-style KV: FlatQuant K2V2 on Llama-2-7B 6.66 vs QuaRot 9.23 (FP16 5.47); Llama-3-8B K4V4 6.20 vs 6.14 [V]. Newer 2025-26 2-bit KV papers: RotateKV (2501.16383), OTT outlier-token tracing (ACL 2025: +3-8 points at 2 bit vs KIVI-like) [S], OSCAR, PatternKV, Kitty [S titles].
- Quantization Hurts Reasoning?: 4-bit KV (QuaRot recommended) near-lossless (<=1% drop) for reasoning models [V].
- Practice [S, blog]: FP8 KV is the default 2026 production choice on H100+, ~50% memory cut with ~0.3-0.7 point long-context regression [S, low-quality source].

## 6. Cross-cutting benchmark facts worth saying aloud
1. Weight-only 4-bit (g128) is "free" (~1% loss) for >=8B models; weight+activation 4-bit still costs 2+ points on small models and more on hard tasks; 8-bit (FP8) is lossless [V: 2411.02355, 2504.04823, 2410.09426].
2. Reasoning models are more fragile: harder tasks (AIME-120) suffer up to 4x greater degradation than GSM8K; RL-trained reasoning models more sensitive than distilled ones [V, 2504.04823 findings 3-4].
3. Models trained on more tokens quantize worse (Llama-3 vs Llama-2; Qwen3 vs Llama-3) [V, 2502.13178 and 2505.02214 attribute to quantization scaling laws].
4. Rotations matter and transfer across formats, but their benefit depends on group size (NVFP4 group 16 loses it) [V, 2509.23202].
5. Calibration cost spectrum: data-free (HIGGS, RTN+Hadamard) / minutes (PrefixQuant 1 min, TurboQuant ~0, AWQ minutes [I]) / ~1 GPU-hour for 8B (GPTQ/GPTAQ 0.2-0.3 h; FlatQuant 0.9 h; SpinQuant 0.5 h) / ~3.5-6 h for 70B (SpinQuant 3.5 h; FlatQuant 5.9 h; GPTAQ 2.7 h; SignRoundV2 2.5-6 h) / 270-336 GPU-h for 70B VQ with FT (QuIP#, AQLM) [V from the papers cited].

## 7. Open items not verified (do not state as fact)
- GPTQ, SmoothQuant, AQLM, QuIP# original-paper numbers (not re-read; used only via later papers).
- OSTQuant W4A4 numbers; DuQuant own tables; AWQ own tables.
- Any claim of a "leaderboard" for PTQ: none found.
- 2026 arXiv entries (ReSpinQuant, TwinQuant, MosaicQuant, D2Quant, ScaleSweep, etc.): abstract/snippet only.
- KronQ and TurboBoA are single-group 2026 papers with self-selected baselines.


---

# State of the art in training-free pruning / sparsity / low-rank / depth / MoE compression (as of Oct 2026)

Tags: [V] read in arXiv HTML full text (fetched through a summarising tool, so numbers are as relayed by it; spot-check before quoting on air); [A] abstract / search snippet only; [S] secondary source; [I] my inference. Numbers not found are not given.
Caveat that applies everywhere: papers use different "compression ratio" definitions (SliceGPT counts slicing % of embedding dim incl. adapter overhead; SVD papers count parameter ratio; pruning papers count removed params excluding embeddings) and different calibration sets, so SliceGPT's WikiText-2 ppl at "20%" on LLaMA-2 7B appears as 7.68 (MoDeGPT paper) and 8.24 (FLAT-LLM paper). Only same-table comparisons are safe.

## 0. Bottom line
1. At equal memory, 4-bit weight-only quantization (GPTQ/AWQ) beats every training-free pruning/low-rank method on perplexity and most tasks [V/A: LLMCBench, UniComp, Zhou et al. EMNLP-Findings 2025]. Note a 4-bit model is ~75% smaller than FP16; to match, pruning/SVD must remove 75%, where they collapse. Honest framing: pruning wins only where (a) hardware has no low-bit kernels, (b) you stack it with quantization, or (c) you need FLOP/latency reductions from dense smaller matrices.
2. Training-free regime leaders by family, with caveats: unstructured: Wanda/SparseGPT baselines, improved by OWL (non-uniform layers), DSnoT, ALPS, RIA (strongest at 60-70% sparsity, but no real speedup); semi-structured 2:4: SparseGPT/ALPS/RIA+channel-permutation (real but hardware-dependent speedup); structured width: MoDeGPT > SliceGPT, FLAP; DISP-LLM claims to beat SliceGPT but uses a learned (not training-free) selection; low-rank: SVD-LLM V2 (training-free) and Dobi-SVD (8 GPU-hours of tiny training on 224 truncation params; not strictly training-free) lead; depth: BlockPruner/SLEB/ShortGPT family, with magnitude compensation (Prune&Comp) improving; MoE: REAP (pruning) beats merging on generative tasks.
3. Reasoning/generation degrade much more than multiple-choice knowledge under all pruning [V: UniComp; V: REAP]. Perplexity-based leaderboards flatter compression.

## 1. Unstructured / semi-structured (2:4)
- SparseGPT (Frantar & Alistarh, ICML 2023, arXiv 2301.00774) [V]. OBS-style layer-wise sparse regression, Hessian-inverse reuse across rows; one-shot, no fine-tuning. OPT-175B 50%: ppl 8.21 vs 8.35 dense; 2:4: 8.74 (+0.39). BLOOM-176B 50%: 8.30 vs 8.11. ~4 h on one A100 for 175B. Speedups: CPU DeepSparse 1.82x at 50% (OPT-2.7B); GPU 2:4 1.54-1.79x on individual OPT-175B layers (layer-level, not end-to-end).
- Wanda (Sun et al., ICLR 2024, arXiv 2306.11695) [V]. Metric |W_ij|*||X_j||_2, per-output-row comparison, no weight update. LLaMA-7B 50%: dense 5.68, magnitude 17.29, Wanda 7.26, SparseGPT 7.22. ~300x faster to prune than SparseGPT. 2:4 on LLaMA-65B: 1.6x matmul speedup in linear layers.
- Why 2:4 speedups are hardware-dependent: needs NVIDIA Ampere+ sparse tensor cores and cuSPARSELt-style kernels; speedups reported are on linear layers/matmuls, not end-to-end [V: Wanda, SparseGPT]; SLEB paper says 2:4 shows minimal speedup at realistic batch sizes [V]; LLMCBench notes 2:4 lacks library support in vLLM/MLC-LLM [V], UniComp finds unstructured gives minimal runtime gains and only 2:4 helps [V]; unstructured 50% gives speedup only on CPU engines like DeepSparse [V]. Also 2:4 is 50% of weights but metadata overhead means memory saving < 50% [I].
- OWL (Yin et al., ICML 2024, arXiv 2310.05175) [V]: non-uniform layerwise sparsity, lower sparsity for layers with more activation outliers. LLaMA-7B 70% WikiText ppl: Wanda 85.77 -> 24.55 with OWL; SparseGPT 26.30 -> 19.49.
- DSnoT (Zhang et al., ICLR 2024, arXiv 2310.08915) [V]: training-free "fine-tuning" via iterative prune-and-grow minimising reconstruction error, no backprop; LLaMA-7B 70%: Wanda 88.84 -> 62.05; ~4.3 s on an A100.
- RIA + channel permutation ("Plug-and-Play", Zhang et al., ICLR 2024) [A]: relative importance and activations metric; channel permutation to preserve important weights under N:M. Exact arXiv id not confirmed; no numbers read.
- ALPS (Meng et al., NeurIPS 2024, arXiv 2406.07831) [V]: l0-constrained ADMM + PCG weight refinement. LLaMA3-8B 70%: 29% lower WikiText ppl, +8% zero-shot vs competing methods (the paper's own wording as relayed).
- Newer (2025-26), [A] only: R2-DSnoT, SparseFW (Frank-Wolfe, reports up to 80% lower per-layer error than Wanda), AutoPrune, F-Wanda (Fisher-reweighted row budgets, arXiv 2608.00481), Symmetric Pruning (arXiv 2501.18980), UniPruning (2510.03291), Z-Pruner (2508.15828). All improve on Wanda; none verified. SparseLLM not searched.
- Reasoning: unstructured Wanda/OWL retains test-time-scaling performance while ShortGPT-style depth pruning breaks it (Monjur et al. 2026, arXiv 2604.25098) [V].

## 2. Structured width
- SliceGPT (Ashkboos et al., ICLR 2024, arXiv 2401.15024) [V]: calibration set 1024x2048 recommended (>=128 OK); slices in 1-3 h on one GPU. At 25% slicing it beats SparseGPT 2:4 on ppl for LLaMA-2 70B (4.60 vs 4.98) and OPT-66B (9.68 vs 10.22); LLaMA-2 70B 25%: 69.75 vs 76.57 zero-shot avg dense. Throughput: H100 1.55x at 25%; RTX6000 16-17% per-token speedup, A100 11-13%. Adds D x D adapter matrices on residual; DISP-LLM says 5-13% extra params [V].
- FLAP (An et al., AAAI 2024, arXiv 2312.11983) [V]: fluctuation metric + bias compensation + adaptive global structure, no retraining, 3-5 min. LLaMA-7B 20%: ppl 14.62 (vs LLM-Pruner 19.77, no FT); zero-shot avg 62.08; 50%: ppl 31.80, avg 50.37. Third-party table (FLAT-LLM paper) LLaMA-2-7B 20%: FLAP 6.76 ppl/53.07 avg vs SliceGPT 8.24/46.26 [A].
- MoDeGPT (Lin et al., 2024, arXiv 2408.09632) [V]: modular decomposition. Nystrom approximation for MLP up/down, CR decomposition for Q/K, SVD for V/O; no adapters, no backprop; 128x2048 calibration (WikiText-2 or Alpaca). WikiText-2 ppl LLaMA-2 7B: 20%: 6.16 vs SliceGPT 7.68; 30%: 7.51 vs 10.47. 13B 30%: 6.10 vs 8.68. Zero-shot LLaMA-2 7B 30%: 60.78 vs 53.73; LLaMA-3 8B 25% (Alpaca calib): 63.96 vs 46.17. LLaMA-2 70B 30%: ppl 5.73 vs 5.76 but zero-shot 66.79 vs 52.65. Cost: 8h26m on one A100 for 13B. Throughput claim is qualitative in what I read.
- DISP-LLM (Gao et al., NeurIPS 2024, arXiv 2410.11988) [V]: dimension-independent structural pruning; each layer selects its own subset of embedding features via index selection/addition, breaking the shared-residual-dimension constraint SliceGPT inherits; no extra params, no weight updates, but the selection masks are learned by gradient (so not strictly training-free; optional LoRA). LLaMA-2 7B WikiText ppl: DISP-LLM 50% 9.84 vs SliceGPT 30% 12.51; zero-shot 30%: 59.53 vs 51.50. LLaMA-2 13B 50%: 7.11 vs SparseGPT 2:4 8.32. Throughput 1.08-1.50x on LLaMA-13B, index ops add overhead.
- SlimGPT (Ling et al., 2024, arXiv 2412.18110) [V]: OBS-style layer-wise structured pruning with grouped Cholesky batched greedy pruning and incremental (non-uniform) per-layer ratio; 20% of LLaMA-7B pruned in ~18 min. LLaMA-7B 50% no FT: ppl 38.83 / avg 52.23 vs LLM-Pruner 40.64/48.35. Compared against LLM-Pruner family, not SliceGPT, in what I read.
- Newer, [A]/[S]: FLAT-LLM (arXiv 2505.23966, [V]): head-wise PCA of value/output, IPRS rank allocation, training-free; LLaMA-2 7B 20%: ppl 6.70 vs SliceGPT 8.24, avg acc 55.16 vs 46.26, speedup 1.04-1.23x. LILA (2609.11163, [V]): calibration-free, spectral; beats WikiText-calibrated SliceGPT by up to 6.0 pp zero-shot on LLaMA-2-7B at 25%, speedup up to 1.64x. Sniper (Goel et al., arXiv 2608.12953, [V]): depth+width knapsack; LLaMA-3.1-8B-Instruct 25% no FT: SliceGPT 62.06 vs Sniper 77.03 retained; 35%: 56.74 vs 64.96; says SliceGPT gives minimal real speedup. CoCurve (2607.17568) [A]: training-free curvature structured pruning; claims over SliceGPT/FLAP/MoDeGPT, no numbers read. HARP (2507.01900) [A], DuoGPT (2506.20194, claims +9.17% accuracy over structured methods at 1.39x iso-speedup) [A]. IrekoGPT (AImageLab, NeurIPS 2026 AXIOM workshop) [S]: fork of SliceGPT, keeps projections to give slimmable nested widths; gradient-free ridge regression correction; claims gains over PCA slimming at high ratios.
- Contrast only (needs retraining): LLM-Pruner (LoRA, 3 h, 50K samples, LLaMA-7B 20% retains 94.97% [V]), Sheared-LLaMA (50B tokens, prunes hidden dim via masks too [V]), Minitron (distillation, up to 40x fewer tokens than scratch [V]).

## 3. Low-rank
- FWSVD (Hsu et al., ICLR 2022) [S]: Fisher-weighted SVD, needs gradients/fine-tune; superseded.
- ASVD (Yuan et al., 2023, arXiv 2312.05821) [V]: activation-scaled SVD + sensitivity-based per-layer rank search; training-free; 10-30% param reduction; KV-cache 50% via low-rank K/V.
- SVD-LLM (Wang et al., ICLR 2025, arXiv 2403.07378) [V]: truncation-aware whitening (Cholesky of XX^T) so dropped singular value = exact loss; sequential LoRA-style update optional. LLaMA-7B 20%: ASVD 11.14 vs SVD-LLM 7.94 (whitening only) vs 7.73 (full). At 7B-class beats SliceGPT/LLM-Pruner per paper [V].
- SVD-LLM V2 (Wang et al., NAACL 2025, arXiv 2503.12340) [V]: per-matrix ratio from theoretical truncation loss + loss-optimised truncation; no retraining, 256 WikiText-2 samples. 20% ratio: LLaMA-7B 7.12 (v1 7.94); LLaMA-3 8B 8.01 (dense 6.14; v1 11.82; ASVD 17.55). At a 7GB budget 28% lower ppl than SliceGPT/BlockPruner. Speedups on A100 LLaMA-7B: 1.29x@20%, 1.63x@40%, 2.71x@80%.
- Dobi-SVD (Wang et al., ICLR 2025, arXiv 2502.02723) [V]: differentiable truncation positions (tanh mask), IPCA weight update, quantised "remapping" so storage ratio is not penalised. NOT training-free (8 GPU-h, 224 params). LLaMA-7B ratio 0.4 WikiText ppl 9.95 (relayed; search snippet said 9.07 - discrepancy, check version) vs SVD-LLM 53.74, LLM-Pruner ~121.5, SliceGPT ~160.5, ASVD 57,057. Claims first SVD method to beat pruning. Speedups: 3x on A100, 12.4x on Titan Xp at 0.4 (settings unclear).
- ESPACE (Sakr & Khailany, NeurIPS 2024, arXiv 2410.05437) [I, not fetched]: projects activations, not weights; Eigen Attention (2408.05646) low-rank KV [not fetched].
- Newer, [A] titles only: BALF (2509.25136, budgeted activation-aware, fine-tuning-free), ARA adaptive rank allocation (2510.19389), IO-SVD (2605.15626), LACE-SVD (2607.03057), Fermi-function rank selection (2512.03062), Beyond Uniform SVD (2510.19385), FlashSVD (2508.01506 / 2605.08314: inference-kernel/memory focus), global rank+sparsity RPCA (2505.03801; low-rank + sparse). Trend: better rank allocation and error compensation, plus kernels. The recurring practical issue: low-rank saves FLOPs only when rank < d_in*d_out/(d_in+d_out) (about half the dimension for square matrices), so small ratios often give no speedup [I].

## 4. Depth / layer removal
- ShortGPT (Men et al., ACL Findings 2025, arXiv 2403.03853) [V]: Block Influence = 1 - cos(input,output) of hidden state; remove lowest-BI layers; no fine-tuning; ~25% params: retains 86.31% (LLaMA-2 7B), 91.64% (13B), 87.35% (Baichuan2-13B) of baseline score; works with quantization; hurts generation tasks more.
- LaCo (Yang et al., 2024, arXiv 2402.11187) [V]: merge rear layers into preceding via parameter differences; Llama2-13B 30% layers: 85.21% retained; avg score 47.55 vs LLM-Pruner 36.19 vs SliceGPT 34.97 (own protocol); 313 s for SliceGPT vs ~15 s.
- SLEB (Song et al., ICML 2024, arXiv 2402.09025) [V]: iterative removal of blocks by perplexity impact; 20% removal: C4 ppl OPT-6.7B 15.99 vs SliceGPT(25%) 27.35; LLaMA-2-70B 7.31 vs 20.03; 1.27x generation speedup on 70B; speedup scales with removed fraction regardless of batch size.
- BlockPruner (ACL Findings 2025) [A]: separate MHA/MLP sub-blocks, ppl-guided iterative; Sliding-window layer merging (arXiv 2502.19159) [A]: depth methods beat width ones (Wanda-sp, FLAP, LLM-Pruner) in latency/throughput at 20-35%; Prune&Comp (2507.18212) [V]: offline weight-rescale fixes hidden-state magnitude gap, training-free, LLaMA-3-8B 5 layers removed: QA retention 93.19%, ppl roughly halved vs baseline; Data-free attention-layer pruning (2512.20636) [A]; XMerge (2609.02083) [A]; E3-Pruner (2511.17205) [A].
- Third-party at 20% LLaMA-2-7B: ShortGPT ppl 14.31 vs SliceGPT 8.24 [A, via FLAT-LLM table]: depth removal is worse on perplexity but faster in wall-clock.
- Depth pruning harms test-time scaling/multi-step reasoning (arXiv 2510.22228 [A]; 2604.25098 [V]).

## 5. MoE-specific training-free
- NAEE (Lu et al., ACL 2024, arXiv 2402.14800) [A]: expert pruning (enumerate subsets minimising layer output error) + dynamic skipping on Mixtral 8x7B; no numbers read.
- MC-SMoE (Li et al., ICLR 2024, arXiv 2310.01334) [A]: routing-statistics-aligned expert merge + low-rank/sparse compress; "up to 80% memory, 20% FLOPs"; compress stage includes fine-tuning/distillation [I; check].
- HC-SMoE (ICML 2025, arXiv 2410.08589) [A]: retraining-free hierarchical clustering on expert outputs, frequency-weighted merging.
- DERN (EMNLP 2025, arXiv 2509.10377) [A]: drop experts, split into neuron segments, reassign; +5% commonsense/MMLU at 50% expert sparsity vs prior retraining-free.
- REAP (Lasby et al., Cerebras, arXiv 2510.13999 v3, 2026) [V]: saliency S_j = mean over routed tokens of gate_j(x)*||f_j(x)||_2. 7 SMoEs 21B-1T. 25% experts removed: ~1.9% avg coding loss vs >5% for merging; 50%: 6.9% vs >20% (HC-SMoE, M-SMoE); Qwen3-Coder-480B and Kimi-K2 near-lossless (<=2%) on code at 50%. Merging fine on multiple choice (~4% drop at 25%) but fails generation; claims merging has irreducible error from losing independent router control. Needs domain-matched calibration (C4 calibration on code -> 0%). Prunes cleanly into quantised formats.
- Router calibration (arXiv 2603.02217) [V]: retraining-free compression has a router-expert mismatch; a lightweight router KD (router ~0.04% of Qwen3-30B-A3B params, ~2 h) recovers pruning/editing/merging losses. Note: not strictly training-free, but cheap.
- Others [A]: AIMER calibration-free expert pruning (2603.18492), OMP-MoE (2609.31631), ConMoE (2605.29350), MoRA (2610.00367), "How to score experts" (2606.15716).
- Caveat [I]: MoE expert pruning shrinks memory (the actual MoE bottleneck) but not active FLOPs/token.

## 6. Benchmarks / surveys and ranking
- LLMCBench (Yang et al., NeurIPS D&B 2024, arXiv 2410.21352) [V]: 3 sparsity methods (LLM-Pruner, Wanda, SparseGPT) vs 4 quantizers (GPTQ, SmoothQuant, AWQ, OmniQuant). Quantization better overall, in generalisation, inference efficiency, trustworthiness; best-in-hardware: INT4; Wanda cheapest to produce; structured sparsity ~ INT8 for acceleration.
- UniComp (arXiv 2602.09130) [V]: LLaMA-3.1-8B; knowledge retention: Wanda-50% 86%, SparseGPT-50% 89%, AWQ-INT4 100%, GPTQ-INT4 100% (reasoning retention 21%/25%/28%/25%). "Knowledge bias": factual tasks 90-100% retained, reasoning/multilingual/instruction 20-70%. Conclusion: quantization best deployment trade-off; pruning unsuitable except hardware-accelerated 2:4. Task-specific calibration lifts pruned GSM8K by ~50% relative.
- Zhou, Kurz, Zhao, "Revisiting Pruning vs Quantization for SLMs" (EMNLP Findings 2025) [A]: quantization beats pruning on LM, esp. at high compression; gap narrows on knowledge/reasoning tasks (OpenBookQA); Wanda-competitive-with-SparseGPT does not carry to SLMs; recommends AWQ.
- Beyond FLOPs (arXiv 2606.09080) [V, thin]: FLOPs reduction does not guarantee latency reduction; structured methods give more consistent speedups; 2:4 as case study.
- Older surveys: Zhu et al. TACL 2024 (Model compression for LLMs), Wan et al. efficient LLM survey, Park et al. 2401.15347 [S, not read].
- Pruning-vs-quant snippet claims [A, unverified source]: 4-bit max accuracy drop 5.6% vs double-digit losses for 75% sparsity; at 50% compression quantization max drop 1.81% vs pruning up to 16.61%.
- Consistent ranking across the sources I read [I, synthesis]: (1) weight-only 4-bit quantization; (2) mild unstructured/2:4 sparsity with OWL/ALPS-style tuning on memory-agnostic hardware; (3) activation-aware low-rank (SVD-LLM V2) and modular decomposition (MoDeGPT) for ratios 20-30%; (4) SliceGPT/FLAP; (5) depth removal wins latency but loses most accuracy on generation.
- Weak spot in evidence: no single paper I found runs SliceGPT, MoDeGPT, SVD-LLM V2, FLAT-LLM, BlockPruner and quantizers on identical calibration, ratio definitions and hardware; cross-paper SliceGPT numbers vary.

## Honest limits of this note
- Fetches went through a summariser, not raw LaTeX; PDFs were binary so HTML versions were used. [V] means HTML full text consulted.
- Not fetched: ESPACE, Eigen Attention, RIA details, SparseLLM, QuaRot/SpinQuant/DuQuant (out of scope after refocus).
- Dates in 2026 arXiv ids (2602-2610) come straight from search results; I did not independently confirm venue acceptance for them.


---

# Efficiency and systems: which training-free compression actually makes inference faster or cheaper?
(Scope revised mid-task from "SliceGPT efficiency" to "state of the art in training-free compression, late 2026". Date of notes: 2026-10-02.)

Tags: [V] verified in full text (I downloaded the PDF and read/grepped it, or read the source code), [A] abstract / fetch-summary only, [S] secondary (blog, press, search summary), [I] my inference/arithmetic. Nothing below is invented; where a number was not found it says so.
Caveat: WebFetch returns a small-model summary of a page, so anything not marked [V] should be treated as [A]/[S] even where the summary quoted numbers.

---------------------------------------------------------------
## 0. Headline take-aways (for the lecture)
1. Compression ratio is a poor predictor of speed. What matters: (a) does the result run on dense, aligned tensor-core GEMMs; (b) is the phase memory-bound (decode, small batch) or compute-bound (prefill, large batch); (c) is there a mature kernel in the serving engine.
2. Weight-only 4-bit (Marlin-class) is the clear practical winner for decode and small/medium batch: ~3.9x kernel speedup at small batch on A10 (-> ~1.5x at batch 128), up to 2.8x end-to-end in vLLM [V MARLIN]; 2-3x cost-per-query reduction at ~99% accuracy recovery on Llama-3.1 8B/70B [V BF16-or-Death].
3. FP8 (W8A8) is "free" on Hopper/Ada: ~lossless, wins at high batch / async serving [V BF16-or-Death]. NVFP4 on Blackwell: vendor claims up to ~2-3x over FP8 with <1% loss, [S] only.
4. W4A4 (QuaRot-style) gives 2-3.3x PREFILL speedup and ~3.6-3.9x peak-memory saving, but on the research kernel base (RTX 3090) and decode speed was not the headline [V QuaRot]. QServe finds naive INT4 dequant overhead of 20-90% on GPUs and settles on W4A8KV4 for serving: 1.2-3.5x over TensorRT-LLM [V QServe].
5. 2:4 sparsity: real but small (about 1.1-1.5x), and as of March 2026 vLLM REMOVED its 2:4 kernels because nobody used them [V PR #36799 via fetch]. Unstructured: no speedup on stock GPUs.
6. Width pruning/slicing: gives fewer, smaller dense GEMMs, so speedups are real if dims stay hardware-aligned, but are smaller than the parameter cut suggests; at batch 1 it can even be slower if dims become misaligned [V Shortened LLaMA; A Dimensional Misalignment]. Depth pruning is the most reliable pure-pruning speedup at batch 1 (near linear in layers removed).
7. KV-cache compression helps long-context/high-batch serving only; at short context or small batch, quantization overhead can make it slower (KVQuant 0.76-0.92x [S]; QuaRot's 4-bit KV is slower than FP16 up to batch 8 [V]).

---------------------------------------------------------------
## 1. SliceGPT itself: where the speed does and does not come from

Source: Ashkboos et al., ICLR 2024, arXiv 2401.15024v2 (full text read) + microsoft/TransformerCompression code (read rotate.py, slicing_scheduler.py, run_slicegpt.py).

### 1.1 What shrinks [V]
- Rows of every W_in (Wq, Wk, Wv, W1/W_up/gate) and columns of every W_out (Wo, W2/down), plus embedding columns and LM-head inputs get deleted, i.e. the residual width D -> D_small. Paper: "SliceGPT also introduces (structured) sparsity in X: entire columns of X are sliced off ... enhances both the computational complexity (in flops) and data movement".
- Code check (rotate.py): `slice_attention_inputs` cuts `W.weight[:, :new_dim]` for Q/K/V (input dim only); `slice_attention_output` cuts `W.weight[:new_dim, :]` for Wo (output dim only). So Wq/Wk/Wv keep their OUTPUT dimension (n_heads*head_dim) unchanged. [V code]

### 1.2 What does NOT shrink [V code / I]
- Attention score computation (QK^T, softmax, AV): head count and head_dim are unchanged -> unchanged FLOPs and bytes. [I from V code]
- KV cache: K and V are still full n_kv_heads*head_dim per token per layer, so slicing does NOT reduce KV-cache size. [I from V code; the paper does not discuss KV cache at all - I grepped for "KV"/"cache" and found nothing relevant.] The only memory benefit is weights, and the freed memory permits larger batch (the paper's Table 11 uses exactly this).
- The paper's own explanation of the 70B 50% result: fewer GPUs needed because weights fit, not because activations shrink.

### 1.3 Shortcut matrices [V]
- Residual path gets a new linear map Q_{l-1}^T Q_l (one per block boundary; both rows and columns deleted so it is D_small x D_small). Paper: "these additional operations cannot be pre-computed and add a small (D x D) overhead ... Nonetheless, they are needed ... and we see real speedup overall". (There are two per block in the code: attn_shortcut_Q and mlp_shortcut_Q; code attaches `self.*_shortcut_Q`.)
- [I] Per-token cost of one D_s x D_s shortcut vs the block: D_s^2 weights vs ~ (4 D^2 attn + 3*D*D_ff MLP). For Llama-2-7B (D=4096, D_ff=11008) at 25% slicing D_s~3072: each shortcut ~9.4M params, two of them ~19M vs ~ 200M per block, i.e. roughly 9-10% of the sliced block weights. Not trivial; it eats a visible chunk of the gain. (Arithmetic mine; not in paper.)

### 1.4 Paper's measured numbers [V]
Kernel-level (Appx A.7, Table 12, A100 80GB, seq len 2048, cuSPARSELt for 2:4, dense PyTorch for slicing; sum of the 5 projection GEMM groups only - no attention scores, no shortcut GEMMs):
| Model | Dense (ms) | 2:4 SparseGPT | SliceGPT 25% | SliceGPT 50% |
| Llama-2 7B | 3.99 | 2.70 (1.48x) | 2.99 (1.33x) | 2.06 (1.94x) |
| Llama-2 13B | 5.93 | 3.95 (1.50x) | 4.57 (1.30x) | 3.11 (1.91x) |
| Llama-2 70B | 16.13 | 12.20 (1.32x) | 12.20 (1.32x) | 8.63 (1.87x) |
Perplexity (WikiText-2) in same table: 7B dense 5.47, 2:4 8.69, slice25 7.24, slice50 17.17; 70B dense 3.32, 2:4 4.98, slice25 4.60, slice50 8.86.
Note: the 25% slice speeds up GEMM time by ~1.30-1.33x, close to 1/0.75 [I]. The 2:4 numbers are GEMM-only microbenchmarks; the authors say end-to-end 2:4 was "not feasible ... at the time of writing".

End-to-end per-token time, batch 1, 128 tokens, HF Transformers, no continuous batching/sharding (Table 2):
- A100 40GB: Llama-2 70B dense 125 ms on 4 GPUs (500 GPU-ms) -> 25% sliced 110 ms on 3 GPUs (330 GPU-ms) = 66% of cost; OPT-66B 114 -> 102 ms, 4 -> 3 GPUs.
- Quadro RTX6000 24GB: 70B dense 252 ms on 7 GPUs (1764 GPU-ms) -> 215 ms on 5 GPUs (1075 GPU-ms) = 64% of cost.
- Latency speedup only 11-13% (A100) and 16-17% (RTX6000); the big win is fewer GPUs. [V]
- Contrast with the 1.3x kernel number: latency gain (1.14x on 70B/A100) is much smaller [I]: shortcut GEMMs, attention-score time and KV traffic unchanged, multi-GPU communication, HF overhead.

Throughput on 80GB H100, seq len 128, max over batch size (Table 11):
- Llama-2 13B (1 GPU, batch 512): 2707 -> 2878 tok/s at 25% (1.06x); 3122 (1.15x) at 50%.
- Llama-2 70B: dense 2 GPUs batch 128: 541 tok/s; 25% slice 2 GPUs batch 256: 839 (1.55x); 50% slice 1 GPU batch 128: 1014 (1.87x per-model; paper says 3.75x "for a fixed number of GPUs").
- OPT-66B 50%: 141 -> 441 tok/s (3.13x; 6.26x per fixed GPUs). OPT-13B: 1.13x/1.22x.
- Interpretation [I]: where weights are a small share of memory (13B on 80GB) slicing barely helps throughput; wins come from batch-size headroom and fewer GPUs for 66-70B.

Compression cost (Table 3, 30% slice, Alpaca calibration, 1024 seqs x 2048 tokens) [V]: Llama-2 7B 44 min on 1xH100; 13B 1h08m on 1xH100; 70B 3h31m on 1xH100 (80GB). Recovery fine-tuning adds 23 min / 44 min / 1h35m on 4xH100. Main text says PCA for 70B takes ~3.5 h on a single H100; eigendecompositions are done in float64 (single precision visibly hurts accuracy; Appendix A.2). Code does the weight rotation matmuls also in float64 [V code].

### 1.5 Hardware alignment: what the paper does and does not say [V]
- The paper text does NOT discuss tile alignment or rounding to 64/128. I grepped. (The prompt's "paper's own rounding" is not in the paper.) The released code has `--round-interval` ("the best value may depend on your hardware"), default 1, and rounds the new dim DOWN to a multiple of round_interval: `new_embedding_dimension -= new_embedding_dimension % round_interval`. Also a scheduler option `round_interval` in slicing_scheduler.py. [V code]
- Hence unrounded sliced widths (e.g., int(0.75*4096)=3072 happens to be aligned, but 30% -> 2867 is not) are the user's problem. Evidence that misalignment really hurts: see 3.5.

### 1.6 Roofline view [I unless tagged]
- Decode, small batch: memory-bound, time ~ weight bytes / bandwidth. Cutting x% of weight bytes -> ~x% faster, minus unchanged costs: KV reads (grow with context*batch), attention, the shortcut GEMMs' weights, kernel-launch overhead. The observed 11-17% latency gain for 25% slicing at batch 1 is lower than 25% (see 1.4), consistent with these fixed costs [I].
- Prefill / large batch: compute-bound; FLOPs on slicable GEMMs drop ~x%, so same story. Kernel table (1.3x at 25%) matches.
- Where the arithmetic intensity crosses over: QServe computes the A100 W4A16 vs W8A8 crossover at m ~ 78 tokens in flight [V QServe], useful as a feel for when "decode" stops being memory-bound.

---------------------------------------------------------------
## 2. Side-by-side evidence across method families

### 2.1 Weight-only quantization (Marlin class)
- MARLIN (Frantar, Castro, Chen, Hoefler, Alistarh; arXiv 2408.11743; PPoPP 2025) [V text grepped]: ~3.9x kernel speedup vs FP16 at small batch on A10 (ideal 4-bit is ~3.87x w/ group scales), degrading gradually to ~1.5x around batch 128; up to 2.8x end-to-end vs vLLM FP16 at batch 16; Sparse-MARLIN (2:4 + 4-bit) up to 3.2x end-to-end and up to +65% over base Marlin.
- Red Hat/Neural Magic Sparse Llama 3.1 8B (blog) [A/S]: 2:4 + FP8/INT4, single-stream latency 3.0x on A5000/A6000 and 2.1x on A100, of which only ~1.1-1.2x (A5000/A6000) / ~1.1x (A100) came from sparsity; 1.2-1.8x max query rate; 98.4% accuracy recovery; required continued pretraining of 13B tokens on 32 H100 in 26 h - so NOT training-free.
- "Give Me BF16 or Give Me Death?" (Kurtic et al., ACL 2025, arXiv 2411.02355) [V text grepped]: >500k evals on Llama-3.1 (8B/70B/405B), vLLM, A6000/A100/H100. FP8 W8A8 effectively lossless; INT8 W8A8 ~1-3% loss; W4A16-INT avg 99.36% recovery vs 99.75% for 8-bit; W4A16 best for synchronous/low-latency, W8A8 best for async continuous batching; W4A16 cuts cost per query 2-3x and latency 1.5-2.5x for 8B/70B, 5-7x cost reduction at 405B (fits on 4 GPUs vs 16 in BF16). Multi-GPU INT8/FP8 speedups in their tables ~1.4-3.0x depending on config.
- Lossless-format option: DFloat11 (arXiv 2504.11651, "70% size, 100% accuracy") [S, only seen in search listing; not read].

### 2.2 W8A8, W4A8, W4A4 (activation quantization)
- QuaRot (Ashkboos et al., arXiv 2404.00456) [V text grepped]: W4A4KV4, Hadamard rotations (same computational-invariance trick as SliceGPT). Llama-2-70B: <=0.47 WikiText-2 ppl loss, 99% of zero-shot score. RTX 3090: 1.97-2.16x prefill speedup on 7B-class layers, up to 3.33x on 70B (batch 64, seq 2048); peak decode memory 3.63x-3.89x less than FP16. Online Hadamard overhead <=7%. 4-bit KV-cache attention is SLOWER than FP16 at batch<=8, and only up to 1.72x faster at batch>=16 (their Table 15). Prefill is where W4A4 wins (compute-bound); decode gains are memory capacity not latency.
- QServe (Lin et al., arXiv 2405.04532) [V text grepped]: dequant overhead of INT4 on CUDA cores 20-90%; W4A8KV4 + SmoothAttention; throughput vs TensorRT-LLM: Llama-3-8B 1.2x (A100)/1.4x (L40S); Qwen1.5-72B 2.4x (A100)/3.5x (L40S); L40S + QServe beats A100 + TRT-LLM, i.e. ~3x cheaper serving claim. Roofline: W4A16 theoretical win when m<78, W8A8 when m>78 on A100.
- Rotation formats caveat: "Transforms for LLM Quantization: The Great Inversion" (Jokar, arXiv 2608.25188, survey ~200 works to June 2026) [A]: with NVFP4, the mantissa-carrying block scale largely removes the incentive for rotations; MXFP4 still benefits from block-confined rotations. Implication for lecture: rotation-based W4A4 is format-dependent; hardware FP4 formats may make QuaRot-style rotations less necessary.
- NVFP4 on Blackwell: vendor/blog claims up to ~3x throughput vs FP8 (B200), <1% accuracy loss (DeepSeek-R1 MMLU 90.8 -> 90.7), supported in TensorRT-LLM, vLLM and SGLang [S - blogs; not verified in a primary NVIDIA document by me]. Treat the "3x" cautiously (it mixes hardware and format gains).

### 2.3 Sparsity
- 2:4 (SparseGPT/Wanda masks; Ampere+ sparse tensor cores): SliceGPT's own A100 cuSPARSELt kernel numbers 1.30-1.50x on GEMMs (Table 12, 1.1 above) [V]. End-to-end: Sparse Llama ~1.1-1.2x from sparsity alone [A/S]; ProxSparse 1.26x end-to-end on Mistral-7B v0.3 and 1.3-1.35x matmul [S search snippet]; SlideSparse (2N-2):2N decode gains 1.07-1.21x [S search snippet]; SpenseGPT (arXiv 2606.10445) hybrid 2:4+dense MLP gets up to 1.2x end-to-end decode on B200 FP8 (Qwen3-32B, Seed-OSS-36B), and strict 2:4 hurts reasoning accuracy [A]. Pattern: 1.1-1.5x ceiling.
- Deployability: vLLM merged PR #36799 on 2026-03-23 removing the Sparse24 compressed-tensors integration and CUTLASS 2:4 kernels; reason given: "not widely used", maintainer burden, binary size; a maintainer noted "no downloads on the models" [A fetch of PR page]. So 2:4 is effectively NOT deployable in mainstream vLLM today (cuSPARSELt / TensorRT-LLM paths may still exist - I did not verify TRT-LLM or SGLang 2:4 status).
- Unstructured: Flash-LLM (arXiv 2309.10285, "load-as-sparse, compute-as-dense") gets kernel wins mainly at high sparsity (its framing is ~70-90%; SpMM beats Sputnik by 2.9x and SparTA by 1.5x on average) [A/S]; at the 50% sparsity SparseGPT/Wanda deliver there is no stock-GPU speed gain [S]. A 2026 paper "Accelerating GPU Inference of LLMs with Moderately Unstructured Sparse Weight Matrices" (arXiv 2607.08786) exists - I did not read it [listing only].

### 2.4 Depth pruning
- Shortened LLaMA (Kim et al., arXiv 2402.02834; full PDF read) [V]. Batch 1, 12 input + 128 output tokens, HF, H100: LLaMA-7B dense 53.7 tok/s (2.4 s); 20% pruned (5.5B): depth (SLEB / theirs) 66.0 tok/s (1.23x) vs width methods Wanda-sp 41.7, FLAP 40.5, LLM-Pruner 43.2 (all SLOWER than dense). 35% pruned (4.5B): depth 80.1 tok/s (1.49x) vs width 40.5-44.4 (slower than dense). On RTX 3090 (24GB): 25.0 -> 28.4 (20%) -> 37.8 tok/s (35%); width 16-21. Vicuna-13B: 45.5 -> 55.7 (21%) -> 69.7 (37%) tok/s; 13B dense OOMs on the 3090. Deep cuts (Vicuna-7B 80% pruned, 1.5B): 182.5 tok/s, 3.4x, but needs continued pretraining to recover (LoRA fails below ~3.7B; training-free SLEB collapses: PPL 18730 at 1.5B).
- Authors' explanation: width pruning leaves GPU-unfriendly dimensions (FFN hidden sizes not divisible by 8) and doesn't reduce the number of memory accesses/kernel launches at small batch; depth pruning removes whole blocks.
- Quality at moderate ratio, LLaMA-7B 20%: avg zero-shot acc depth (PPL criterion, LoRA) 61.9 vs dense 66.3; SLEB (training-free) 57.6; width methods 51.8-61.8 [V]. Depth pruning's quality edge needs LoRA; training-free SLEB is weaker.
- Training-free depth pruning (SLEB, ShortGPT, etc.) is deployable anywhere with zero engine support (just a smaller config). [I]

### 2.5 Low-rank (SVD-family)
- SVD-LLM V2 [S, search snippet of the paper's tables]: generation speedup 1.29x at 20% ratio, 1.63x at 40%, 2.08x at 60%, 2.71x at 80% (so small ratios barely pay off; two sequential skinny GEMMs per matrix).
- "Why Smaller Is Slower? Dimensional Misalignment in Compressed LLMs" (arXiv 2604.09595; Llama-3-8B, A100-80GB) [A fetch summary]: ASVD at 15% parameter reduction has +1% latency (100.5 vs 99.6 ms) because ~95% of dims are misaligned; LLM-Pruner (17% weights misaligned) +38% latency (137.7 ms); with their GPU-aligned fix (GAC, knapsack over aligned ranks) -33% (67.1 ms) and -12% (88.0 ms), up to 1.5x. Minimum requirement d mod 8 = 0; mod 16/32 better. Quoted penalties: SDPA backend fallback ~90%, FlashAttention-2 template ~30%, cuBLAS ~30%, tensor-core under-utilization ~70%. SliceGPT was not evaluated, but its residual width is exactly the kind of dimension that must be aligned (and non-multiple widths also affect every GEMM and the shortcut). [I]

### 2.6 Width slicing (SliceGPT-type)
See section 1. Relative to the above: 25% slicing = 1.3x GEMM, 1.06-1.55x throughput, 1.11-1.17x latency; WikiText-2 PPL at 25% slicing beats 2:4 on Llama-2 7B (7.24 vs 8.69) and 70B (4.60 vs 4.98) [V Table 12], though both are far above dense (5.47 / 3.32) and 50% slicing is much worse (17.17 / 8.86). Paper concedes: "Smaller but dense LMs perform better than LMs with 13B parameters or less pruned to similar sizes" [V].

### 2.7 KV-cache compression
- KIVI (Liu et al., ICML 2024, arXiv 2402.02750) [A]: tuning-free 2-bit, 2.6x lower peak memory including weights, up to 4x bigger batch, 2.35-3.47x throughput on their workloads.
- Benchmarks: "KV Cache Compression, But What Must We Give in Return?" (arXiv 2407.01527, EMNLP Findings 2024) [A]: 10+ methods, 65 settings, 7 task categories - quality is workload-sensitive. "Benchmarking KV-Cache Optimizations across Task Quality and System Performance" (arXiv 2607.05399) [A]: KIVI4 ~3.2x compression and most stable quality; KIVI2 5.2-5.3x; SnapKV 4x fixed budget with best long-context throughput; TurboQuant (rotation-based) highest latency overhead, big summarization drops (35-36%); headline "compression ratio alone is a poor predictor of end-to-end performance". "Quantization Dominates Rank Reduction for KV-Cache Compression" (arXiv 2604.11501) [S snippet]: at matched storage quantization beats low-rank. KVQuant real-world 0.76-0.92x end-to-end [S, DEV Community post, weak]. QuaRot data point: 4-bit KV slower than FP16 up to batch 8 [V].
- Relevance to slicing: slicing does not touch the KV cache (1.2), so it is orthogonal and composable with KV quantization/eviction, but gives no long-context help. [I]

---------------------------------------------------------------
## 3. Composition
- Quantization + sparsity: proven in kernels: Sparse-MARLIN (4-bit + 2:4, up to 3.2x e2e) [V] and Sparse Llama (FP8/INT4 + 2:4; sparsity share small) [A/S]; but 2:4 path removed from vLLM in 2026 [A].
- Quantization + KV compression: Budget-Aware Compression Pipeline (Yu & Shen, arXiv 2608.30076, GroundLM 2026 / EMNLP workshop) [S search listing; PDF fetch 404 so not read]: 4-bit weights + layerwise pruning + hierarchical KV sparsification + INT8 KV cache took a 70B model to ~33 GB, ~57 tok/s at 10k-token prompts on one A40, within 5% absolute accuracy; finding: layer-wise pruning made weight quantization more robust, KV sparsification complements INT8 KV quant, static vector quantizers conflict with dynamic caching.
- Rotation + slicing: SliceGPT and QuaRot share computational invariance (both by Ashkboos; QuaRot cites it). Slicing needs data-dependent PCA rotation; QuaRot uses random Hadamard. I found NO paper that combines SliceGPT with 4-bit quantization in my searches (searched "SliceGPT quantization 4-bit combined", got only QuaRot/SpinQuant background); the SliceGPT paper itself lists quantization as complementary future work [V]. Absence of evidence only; I cannot assert nothing exists. [I]
- Slicing follow-up: "Output-aware Residual Stream Pruning for LLMs" (arXiv 2609.35579, Sept 2026) [A]: replaces SliceGPT's activation-reconstruction PCA with an output-KL-sensitivity-weighted covariance (eigendecomposition of a sensitivity-weighted matrix; needs two d x d statistics C and H, H the only extra vs SliceGPT; with H=I it reduces to SliceGPT per search snippet). Improves ppl/downstream at several ratios. The fetch gave no speed/shortcut information. Also seen in search listing only: IrekoGPT (aimagelab, "post-hoc slimmable LLMs from structured pruning", NeurIPS 2026 AXIOM workshop) - not read.

---------------------------------------------------------------
## 4. What is actually deployable (serving support) - as of what I could verify
- vLLM [A fetch of docs.vllm.ai quantization page]: AutoAWQ, GPTQModel, bitsandbytes, LLM Compressor (FP8 W8A8, INT8 W8A8, INT4 W4A16), NVIDIA ModelOpt, TorchAO, AMD Quark, quantized (FP8) KV cache, GGUF, online quantization; Marlin kernels serve GPTQ/AWQ/FP8/FP4 and need Turing or newer; AWQ Turing+; GPTQ Volta+; plugin API for custom quant configs. 2:4 sparse REMOVED (Mar 2026). No dedicated support for non-uniform hidden size per layer or SliceGPT shortcut layers known to me; searching "SliceGPT vLLM" found nothing. SliceGPT's repo saves sliced models as .pt with `--sliced-model-path` and requires re-specifying sparsity at load; shortcut Q's are extra parameters `*_shortcut_Q` attached to modules [A fetch + V code]. So serving a sliced checkpoint in vLLM would need a custom model class (shortcut linears, sliced widths, and in the HF-config sense hidden_size != n_heads*head_dim input). [I]
- TensorRT-LLM [A search of its docs]: NVFP4, FP8 per-tensor, FP8 KV cache, NVFP4 KV cache, W4A8 AWQ/GPTQ, W4A16 AWQ/GPTQ, via ModelOpt.
- SGLang [A]: fp8, mxfp4, NVFP4/MXFP4 PTQ export, optimized AWQ kernels, offline pre-quantized weights (GPTQ/AWQ).
- I did NOT verify Machete (Hopper W4A16/W4A8 mixed-precision GEMM) numbers, nor TRT-LLM/SGLang 2:4 status, nor SGLang KV quantization specifics.

---------------------------------------------------------------
## 5. Making SliceGPT cheaper/faster (what is documented vs speculation)
- Documented cost: see 1.4. Dominated by forward passes over the calibration set plus D x D eigendecompositions per layer in float64 (paper: single-precision eigenvectors degrade accuracy) [V]. 70B on a single 80GB H100 in ~3.5 h [V]. Calibration-set ablation: >=128 sequences give sensible results; main runs use 1024 x 2048 [V].
- Shortcut removal / sharing: the paper does not propose it. [V negative]. Ideas (all [I]): (a) choose Q_l = Q_{l-1} (shared Q) -> shortcut is identity, no extra GEMM, at the price of a worse subspace per layer; the paper notes that different layers need different Q ("signals at different blocks were not aligned") [V]; (b) fuse shortcut into the next block's residual add via a fused GEMM+add epilogue; (c) the shortcut is square D_s x D_s and could be low-rank/identity-plus-low-rank because consecutive PCA bases are highly overlapping (hypothesis, untested here). Per-layer non-uniform widths: paper Table 6 shows variable slicing by layer helps OPT PPL (e.g., OPT 13B 11.04 -> 10.76) but hurts Llama-2 (7B 6.84 -> 7.63) [V]; non-uniform widths further complicate alignment and tensor-parallel splitting [I].
- Efficient PCA variants (randomized SVD, streaming covariance): the covariance is already accumulated streaming without materializing the N x D signal matrix (paper: "we never materialize the N x D signal matrix") [V]. I found no published randomized-PCA SliceGPT variant.

---------------------------------------------------------------
## 6. Practical guidance: slicing vs a smaller dense model; cost angle
- Paper's own concession [V]: below ~13B parameters, smaller dense models beat pruned-to-same-size models; they expect that to change. Minitron (NVIDIA, arXiv 2407.14679) [A]: prune (depth/width/heads/MLP) + distill from 15B -> 8B/4B using up to 40x fewer tokens than scratch, up to 16% MMLU better than scratch, 1.8x training-compute savings for family; Llama-3.1-Minitron-4B-Depth reports 2.7x throughput vs Llama 3.1 8B [S search summary]. Sheared-LLaMA (Xia et al., arXiv 2310.06694) [A/S]: 1.3B/2.7B beat equal-size open models with <3% of from-scratch compute. These need real training tokens, so are "cheap training", not training-free.
- At equal parameter count and no retraining, the best training-free speed/quality route at moderate cost is quantization (W4A16/FP8) rather than slicing: e.g., 99% accuracy recovery with 2-3x cost cut [V BF16-or-Death] vs slicing 25%: 96% (70B) / 87% (Phi-2) zero-shot retention, 1.1-1.5x [V SliceGPT]. Slicing's niche: reducing GPU count when weight memory is the binding constraint, orthogonal to quantization (should stack, untested [I]).
- Energy/cost: SliceGPT reports GPU-ms per token (cost proxy): 66% (A100) and 64% (RTX6000) of dense for 70B [V], with the caveat that the HF baseline lacks continuous batching so dense could improve more [V footnote 4]. Quantization cost numbers: 2-3x per-query cost reduction (8B/70B), 5-7x at 405B [V]. QServe claims ~3x serving-cost reduction via L40S vs A100 [V claim in abstract]. I found no joules/token measurement for slicing or other methods; "energy" claims in the literature are proxies via GPU count or time [I].
- Rule of thumb for the lecture [I]: (1) Batch-1 / edge: weight-only 4-bit. (2) High-throughput datacenter on Hopper/Ada: FP8 (lossless) or W4A8KV4 (QServe). (3) Blackwell: NVFP4 (vendor claims). (4) Long context: add KV quantization/eviction, expect gains only past some length/batch. (5) Pruning families: depth pruning is the only reliably faster pruning at batch 1; width/slice/low-rank need alignment and pay shortcut or two-GEMM overheads; 2:4 caps at ~1.1-1.5x and lost vLLM support.

---------------------------------------------------------------
## 7. Open gaps / where efficiency could improve
- Kernels: sliced widths need aligned dims (round_interval) and a fused shortcut; no public engine supports them. Low-bit: Marlin speedup decays to ~1.5x by batch 128 (compute-bound) [V]; W4A4 needs hardware 4-bit tensor cores (Blackwell) to avoid dequant overhead [V QServe reasoning]. Sparse tensor cores: ecosystem retreat (vLLM removal).
- Formats: NVFP4/MXFP4 microscaling reduces the need for outlier-suppressing rotations [A Jokar].
- No benchmark I found measures slicing, 2:4, depth, quantization and KV compression side by side at equal quality on one engine/hardware; closest are Kurtic et al. (quantization only), Kim et al. (pruning families, HF, batch 1), and SliceGPT Appendix A.7 (GEMM-only). A joint apples-to-apples study would be a real contribution. [I]
- Not verified / not found: any SliceGPT + quantization paper; vLLM/SGLang SliceGPT loaders; randomized-PCA slicing; Machete numbers; primary NVIDIA numbers for NVFP4; TRT-LLM and SGLang 2:4 status.

---------------------------------------------------------------
## Sources
- SliceGPT: https://arxiv.org/abs/2401.15024 (PDF read) ; code https://github.com/microsoft/TransformerCompression (src/slicegpt/rotate.py, slicing_scheduler.py, experiments/run_slicegpt.py read)
- MARLIN: https://arxiv.org/abs/2408.11743 (PDF read)
- QuaRot: https://arxiv.org/abs/2404.00456 (PDF read)
- QServe: https://arxiv.org/abs/2405.04532 (PDF read)
- Give Me BF16 or Give Me Death: https://arxiv.org/abs/2411.02355 (PDF read)
- Shortened LLaMA: https://arxiv.org/abs/2402.02834 (PDF read)
- KIVI: https://arxiv.org/abs/2402.02750 (abstract)
- Red Hat Sparse Llama: https://developers.redhat.com/articles/2025/02/28/24-sparse-llama-smaller-models-efficient-gpu-inference
- SpenseGPT: https://arxiv.org/abs/2606.10445
- Dimensional misalignment: https://arxiv.org/abs/2604.09595
- KV benchmarks: https://arxiv.org/abs/2607.05399 ; https://arxiv.org/abs/2407.01527 ; https://arxiv.org/abs/2604.11501
- Budget-aware pipeline: https://arxiv.org/abs/2608.30076
- Output-aware residual stream pruning: https://arxiv.org/abs/2609.35579
- Transforms for LLM quantization: https://arxiv.org/abs/2608.25188
- vLLM Sparse24 removal: https://github.com/vllm-project/vllm/pull/36799 ; vLLM quantization docs: https://docs.vllm.ai/en/latest/features/quantization/
- TensorRT-LLM quant docs: https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html ; SGLang: https://docs.sglang.io/docs/advanced_features/quantization
- Minitron: https://arxiv.org/abs/2407.14679 ; Sheared-LLaMA: https://arxiv.org/abs/2310.06694
- Flash-LLM: https://arxiv.org/abs/2309.10285 ; SVD-LLM V2: https://arxiv.org/abs/2503.12340


---

# Statistical foundations of training-free compression, and where it can be improved

Scope (revised): post-training quantization (PTQ), pruning and low-rank/rotation methods; SliceGPT/PCA is one worked example. Audience: statisticians / ML researchers.

## Tag legend and honesty note
- [V] verified in primary full text. NOTE: in this session I could NOT obtain any full text (the SliceGPT PDF fetch returned binary garbage). So almost nothing is [V]; I have not used the tag for paper claims.
- [A] read from the arXiv abstract page or a search-result abstract this session.
- [S] secondary summary, or a classical textbook/paper result recalled from memory and NOT re-fetched this session. Re-check before quoting numbers.
- [I] my inference or derivation (derivations are checkable by hand; I flag the assumptions).
- Numbers: only those that appear in a cited abstract/summary are given. No invented numbers. Cost estimates are in big-O or "same order as X", not seconds.

## 0. Common notation
Layer: Y = W X, W is m x d, X is d x N (N calibration tokens). Compressed layer W_hat. Second-moment matrix C = X X^T / N (d x d). Layer-wise reconstruction error:

  L(W_hat) = ||W X - W_hat X||_F^2 = tr( (W - W_hat) (N C) (W - W_hat)^T ).

Per output row w (1 x d), the loss is quadratic in the weight error e = w - w_hat: L_row = N e C e^T. So the Hessian w.r.t. a row is H = 2 X X^T = 2 N C (shared across all rows of the layer). [S: Frantar & Alistarh OBS/OBC; GPTQ arXiv 2210.17323 [A]; SparseGPT 2301.00774 [A] "reduces pruning to large-scale sparse regression".]

Statistical reading [I]: this is a linear regression with design X and "response" W X; the "coefficient" is W; the compression constraint (quantized grid, sparsity mask, rank k, column deletion) is a constraint set on W_hat. C is the Gram/design covariance. Everything below is either (a) a better loss than L, (b) a better estimate of C, (c) a better constraint-allocation, or (d) a better solver.

## 1. The layer-wise objective and the OBS/Hessian view

### 1.1 OBS and why GPTQ/SparseGPT work
- OBS (LeCun/Hassibi-Stork): removing weight q with optimal compensation of the rest changes the loss by dL = w_q^2 / (2 [H^-1]_qq) and the compensation is delta_w = - (w_q / [H^-1]_qq) H^-1 e_q. [S: Hassibi & Stork 1993; LeCun et al. 1990 OBD].
- GPTQ: quantizes columns in a fixed order, after each column applies the OBS update to the not-yet-quantized columns; uses a Cholesky factor of H^-1 for stability; claims >3 orders of magnitude speed-up over OBQ [A: abstract of 2210.17323 via search summary]. Dampening: H <- H + lambda * mean(diag H) * I [S; this is the standard implementation detail, a ridge/linear-shrinkage-to-identity. Check the exact default fraction in the code before citing a number.]
- SparseGPT: same Hessian machinery, mask selection by w^2 / [H^-1]_qq, solved column-blocks at a time [A/S].
- Statistical identity [I]: the OBS compensation is exactly the least-squares re-fit of the surviving coefficients on the same X, i.e. the conditional-mean correction of a regression after dropping a regressor. Quantization error is therefore treated as "missing regressors" with the constraint that the refit values lie on the grid (a lattice-constrained least squares / closest-vector-like problem; GPTQ ~ Babai's nearest-plane algorithm, a connection made explicitly in the 2025 literature, but I have not verified a specific citation [I]).

### 1.2 SliceGPT worked example: PCA is the solution of a different, simpler problem
SliceGPT [A: arXiv 2401.15024; text not retrieved]: per block, estimates covariance of residual-stream activations on a calibration set, takes eigenvectors Q (PCA), rotates (computational invariance of RMSNorm transformers), and deletes the d - k directions with the smallest eigenvalues.
- PCA solves: min over rank-k orthogonal projectors P of E||x - P x||^2 = tr((I-P)C). Solution: top-k eigenvectors of C; loss = sum of the d-k smallest eigenvalues.
- Output error when the consumer weight is W (m x d): E||W(x - Px)||^2 = tr( W^T-weighted ) = tr( G (I-P) C (I-P) ), G = W^T W (sum over all consumers of the residual stream, G = sum_j W_j^T W_j; note that in a transformer block the stream feeds Q,K,V and the MLP up/gate, and is also written to by O and MLP down). [I, derived]
- PCA is output-optimal only if G is proportional to I (e.g. W has orthonormal rows) or G and C commute AND the eigenvalue ordering by lambda_j * g_j is the same as by lambda_j. In general PCA ignores G. [I]
- Because the sliced model must remain orthogonally invariant for RMSNorm, the projector must stay ORTHOGONAL. The unconstrained output-aware optimum is oblique: P = G^{-1/2} U_k U_k^T G^{1/2} with U_k top eigenvectors of G^{1/2} C G^{1/2}. This breaks RMSNorm commutation, so for SliceGPT one must optimise tr(G P_perp C P_perp) over orthogonal P instead (non-convex; see section 6). [I, derived; the oblique solution is the reduced-rank-regression solution, see 2.3]
- In-sample consequence (important, easily missed) [I, derived]: in the PCA basis the sample covariance is exactly diagonal, so regressing the removed coordinates on the kept ones in-sample yields ZERO coefficients. Hence an OBS/SparseGPT-style least-squares "compensation" of the kept weights gives NO in-sample gain for PCA deletion computed from the same C. The compensations that can help are (i) out-of-sample (the generalisation gap), (ii) regression against the DENSE model's output rather than the sliced model's input (section 3), (iii) the mean (section 9). Contrast: SVD-LLM's update step has a different purpose (section 2.2).

## 2. Output-aware objectives; which are cheap layer by layer

### 2.1 Weighting by the consumer (weighted low-rank approximation)
Generic form: min over constrained W_hat of tr( (W - W_hat) S (W - W_hat)^T ), S a d x d PSD weight on the input side (and optionally a left weight on the output side). Choices of S:
- S = I: plain SVD / magnitude (ignores data).
- S = diag(mean |x_j|^alpha) : ASVD activation-aware scaling; ASVD adds an iterative per-layer rank-sensitivity calibration; abstract claims 10-20% parameter compression without losing reasoning capacity [A: arXiv 2312.05821].
- S = C (full second moment): exact layer-output error. Solved by whitening, 2.2.
- S = Fisher / Gauss-Newton: 2.4.
Cost: all need one forward pass over calibration data and an accumulate of X X^T (O(N d^2) flops, d^2 memory per distinct input; d = 4096 gives about 16.8M entries (67 MB in fp32) per matrix; no invented timings). Cheap: same order as the SliceGPT/GPTQ calibration pass.

### 2.2 Whitening (SVD-LLM) = exact solution of the reduced-rank problem
Let C = S_c S_c^T (Cholesky, S_c lower triangular). Then ||(W - W_hat) X||_F^2 = ||(W - W_hat) S_c||_F^2 (x sqrt(N)) when C is the sample matrix. Eckart-Young on W S_c: W S_c = U Sigma V^T, keep top k, W_hat = U_k Sigma_k V_k^T S_c^{-1}. The truncation loss is then exactly sum_{i>k} sigma_i^2: "singular value = compression loss" (the "truncation-aware" property). SVD-LLM [A: arXiv 2403.07378, ICLR 2025] does exactly this (whitening matrix from Cholesky, SVD of W S) and adds a "parameter update with sequential low-rank approximation" [A].
- Statistical identity [I]: this is reduced-rank regression with known coefficient (Izenman 1975; Anderson 1951) under the C-metric, and equals canonical-correlation-type problems when the target is a separate Y: the optimal rank-k map is C^{-1/2} [C^{1/2} B_ols]_k [S: Izenman 1975; Reinsel & Velu].
- Failure modes tied to estimation [I]: needs C^{-1} (Cholesky of an ill-conditioned or rank-deficient estimate; N < d breaks it). Hence need shrinkage/dampening (section 4). A perturbed whitening is also unstable because S_c^{-1} amplifies errors in the small-eigenvalue directions, exactly where the estimate is noisiest.
- Cost: Cholesky O(d^3) + SVD of W S_c O(m d min(m,d)) per matrix. Same order as GPTQ's Cholesky. Cheap.

### 2.3 Reduced-rank regression and CCA views
- Target Y = W X. Rank-k map M minimising ||Y - M X||: (RRR) M = [Y X^T (X X^T)^-1 X Y^T ... ] top-k eigenvectors of Sigma_YX Sigma_XX^{-1} Sigma_XY projected back [S: Izenman 1975]. With Y = W X the OLS fit is W itself, giving 2.2.
- CCA / "sufficient dimension reduction" view: the residual-stream subspace worth keeping is the one predictive of the downstream OUTPUT (next-layer activations or logits), not of itself. CCA between x and the next layer's pre-activation y: directions maximise correlation, whitening both sides. [S/I; I do not know of a paper applying CCA directly to LLM slicing.]
- Cost: needs cross-covariance Sigma_XY (d x m) in addition; for an LLM block m ~ d or 4d, same order as C. Cheap per layer.

### 2.4 Fisher / loss-weighted and second-order objectives
- FWSVD (Hsu et al., ICLR 2022): weighted low-rank factorization with row weights from the (diagonal) Fisher information; abstract-level claim: reconstruction error is larger than SVD's but task accuracy much closer to original [A: arXiv 2207.00112].
- Loss-aware second-order form [I, derived]: delta L ~ 0.5 * E[ dx^T H_x dx ] with dx = (I-P) x and H_x the Hessian of the true loss w.r.t. the residual stream activation. Gauss-Newton/Fisher approximation H_x ~ E[g g^T], g = dLoss/dx. Hence delta L ~ 0.5 * tr(H_x (I-P) C (I-P)). Same quadratic form as 1.2 with G replaced by H_x, i.e. both sides of the problem are "covariance of x" and "covariance of loss-gradient w.r.t. x".
- Kronecker-factored (K-FAC) version: H ~ H_X (x) H_G with H_G the covariance of back-propagated gradients: KronQ (Lee, Li, Yin, Panda) uses H ~ H_X (x) H_G, "bidirectional incoherence processing with activation and gradient covariances" and a new sensitivity metric for mixed precision; abstract reports 7.93 perplexity on WikiText-2 for 2-bit LLaMA-3-70B [A: arXiv 2607.07964].
- Caveats [S]: the empirical Fisher is not the Fisher (Kunstner et al., NeurIPS 2019); with finite calibration data and a pretrained model near a minimum, E[g] ~ 0 but E[g g^T] is a noisy low-N estimate with heavy tails. Needs a backward pass: cost = forward + backward over the calibration set plus accumulating a d x d gradient covariance per layer; memory for all layers' gradient covariances is the real constraint, so do it streaming layer by layer.
- Ranking by cost [I]: (1) data-aware diagonal (ASVD/Wanda-like): trivial; (2) full C whitening/OBS (SVD-LLM, GPTQ, SparseGPT): one forward pass + O(d^3) per matrix; (3) Fisher/K-FAC: one backward pass, 2-3x the data pass cost, extra d^2 per layer; (4) full Hessian/learned objectives: not layer-by-layer cheap.

## 3. What the layer-wise objective ignores: error propagation and the loss gap

### 3.1 Cross-layer accumulation (asymmetric calibration)
Standard GPTQ calibrates layer l on inputs X_q coming through already-quantized earlier layers but fits against the quantized-input output W X_q, or (variants) both X from the quantized model and the target from the same path. The error of earlier layers is then "absorbed" as if it were the signal. Asymmetric calibration fits the DENSE output on the QUANTIZED input:

  min over W_hat on grid:  ||W X_fp - W_hat X_q||_F^2,

whose minimiser adds a cross term: expanding, H = X_q X_q^T and the linear term is W X_fp X_q^T, so GPTQ's update picks up a residual R = (X_fp - X_q) X_q^T term. [I, derived]
- GPTAQ (Intelligent Computing Lab, ICML 2025): "matches the quantized layer's output to the exact output in the full-precision model (asymmetric calibration)", closed form via optimal brain compression, with channel parallelisation, neuron decomposition, Cholesky reformulation; reports lower perplexity and cumulative error at 2-4 bits than GPTQ [A: arXiv 2504.02692].
- "Rethinking Residual Errors in Compensation-based LLM Quantization" (Li et al., ICLR 2026): even GPTAQ aligns to the output of the COMPENSATED weights rather than the true dense output within a layer; they add a "compensation-aware error" and align to the original model's output [A: arXiv 2604.07955].
- Related earlier lines: sequential/ block-wise reconstruction (AdaRound/BRECQ, [S]); LoaQ "layer-wise output approximation quantization" [A: 2509.06297, title only]; "Augmenting Hessians with inter-layer dependencies for mixed-precision PTQ" [A: 2306.04879, title only].
- Statistical reading [I]: this is "errors-in-variables / covariate-shift" regression: the training design is X_q but the target relation is fixed by X_fp. Using X_fp on both sides (symmetric, dense-input) ignores the shift; using X_q on both sides absorbs it into the fit and propagates bias. The asymmetric objective is the right "predict the dense output from what the compressed network will actually see" estimator.
- SliceGPT connection [I]: SliceGPT computes Q for layer l from activations that depend on earlier slicing [S: I believe the official procedure runs calibration through the already-modified model; verify in the code]. The proper correction is a regression of dense output on sliced input (the W' update), not an OBS-in-PCA-basis update (which is zero in-sample, 1.2).
- Cost: needs the dense activations kept (or recomputed): memory x2 for activations, one extra forward; solver cost about GPTQ + a d x d cross-moment X_fp X_q^T.

### 3.2 Gap to the true loss
Sum of per-layer reconstruction errors is not the loss. Reasons [I]: (a) errors in different layers interact (non-additive, anti-correlated corrections occur since later layers can partly undo earlier errors: "perturbation absorption", e.g. 2606.15161 title-level snippet [A]); (b) the nonlinearity and RMSNorm after the layer rescale error per token; (c) the final loss depends on the output through a Jacobian whose singular directions are not the activation covariance's. Remedies: Fisher/K-FAC weighting (2.4), block-wise or end-to-end distillation losses (training-based, outside "training-free"), LoaQ-style output approximation.
- Token reweighting by RMSNorm [I]: the consumer sees x / rms(x). Tokens with huge norm (attention-sink tokens, see 5) matter little in relative error after normalisation, but dominate the raw covariance C. A consumer-aware C is C_norm = E[ x x^T / rms(x)^2 ] (a weighted second-moment); for SliceGPT's deleted directions the deletion error persists additively in the stream, so the right weighting depends on position in the network (pre-norm blocks see normalised error at every consumer, but the stream error accumulates). Experiment to run: compare PCA of C vs C_norm on held-out loss. Cost: identical to PCA (weights per token).

## 4. Estimating C (or H) from small, dependent calibration sets

### 4.1 Effective sample size
- Tokens inside one sequence share context; the sequence is the natural independent unit (cluster). If we have S sequences of length T and within-sequence correlation of the relevant statistic is rho, the design effect is about 1 + (T-1) rho and N_eff ~ S T / (1 + (T-1) rho), between S (rho = 1) and S T (rho = 0). [I, standard cluster-sampling formula, e.g. Kish design effect [S]]
- For stationary AR(1) the effective size of a MEAN is N (1-rho)/(1+rho) [S]. For a COVARIANCE entry the relevant series is x_i x_j, with its own (typically larger for heavy tails) autocorrelation, so the second-moment ESS is not the mean ESS [I]. Never quote token count as N when judging gamma = d/N.
- Consequence [I, central]: whether we are in the regime gamma = d/N small (all fine) or gamma ~ O(1) (spiked-model trouble) hinges on whether N_eff is closer to number of sequences (~10^3) or number of tokens (~10^6). With d = 4096: gamma ~ 4 vs gamma ~ 0.002. These are the only numbers here; they are arithmetic on d = 4096, 1000 sequences x 2048 tokens (the lecture's example), not measurements.
- Measure it rather than assume: ESS via batch-means or cluster-bootstrap of the statistic of interest (retained variance, subspace angle); see section 8.

### 4.2 Random matrix theory: what is reliably estimated
Spiked covariance model: Sigma = I + sum_i (ell_i - 1) v_i v_i^T (noise floor 1). gamma = d / N_eff.
- Marchenko-Pastur bulk of sample eigenvalues: support [(1 - sqrt(gamma))^2, (1 + sqrt(gamma))^2] times the noise variance. [S: Marchenko-Pastur 1967]
- BBP phase transition (Baik-Ben Arous-Peche 2005): a spike with ell > 1 + sqrt(gamma) separates from the bulk; below the threshold the sample eigenvector is asymptotically orthogonal to the true one. Above: sample eigenvalue -> ell (1 + gamma / (ell - 1)) (biased UP), and the squared cosine between sample and population eigenvector -> (1 - gamma/(ell-1)^2) / (1 + gamma/(ell-1)) (<1: inconsistent). [S: Paul 2007; Benaych-Georges & Nadakuditi 2011; check formula before quoting.]
- Implications for compression [I]:
  (i) Only components with signal-to-noise above 1 + sqrt(gamma) are identifiable; the cut point k that discards near-bulk eigenvalues is in the unidentifiable region, so which subspace is kept there is partly noise, but the VARIANCE cost of using the wrong vectors there is small because those eigenvalues are almost equal (gap-free bound, section 6).
  (ii) Sample eigenvalues in the tail are biased down and the head biased up: in-sample "retained variance" is optimistic. Use sample-split retained variance tr(Q_k^T C_B Q_k) with Q_k from split A as an honest estimate.
  (iii) LLM activation spectra are roughly power-law without a clean noise floor, so spiked-model thresholds (BBP, Gavish-Donoho) describe where noise is expected to dominate, not a true generative model. Treat as heuristic [I].
- Population-spectrum-aware estimators: Ledoit-Wolf nonlinear shrinkage (2012, 2020), optimal eigenvalue shrinkage for spiked models (Donoho, Gavish, Johnstone, Ann. Stat. 2018) [S]. Both change EIGENVALUES, keep sample EIGENVECTORS (rotation-equivariant). Relevant for whitening and dimension choice, not for which vectors SliceGPT keeps.

### 4.3 Shrinkage and dampening (the key subtlety)
- Ledoit-Wolf 2004: S_shrunk = (1 - a) S + a mu I with mu = tr(S)/d; oracle-approximating shrinkage (OAS; Chen, Wiesel, Eldar, Hero 2010) is the same form with a different a, better for small N Gaussian. [S]
- Fact [I, trivial but crucial]: shrinking toward a multiple of the identity leaves ALL eigenvectors unchanged and moves eigenvalues toward their mean. So it cannot improve the PCA subspace used by SliceGPT; it improves inverses and condition numbers: exactly what Cholesky whitening (SVD-LLM) and the GPTQ/SparseGPT inverse Hessian need. GPTQ's "dampening" H + lambda I is this shrinkage in disguise [S; I did not verify the exact default constant].
- To change eigenvectors the target must be non-isotropic: diagonal target (Schafer-Strimmer style [S]), a pooled-corpus covariance, a "structured" Kronecker/low-rank-plus-diagonal target, or a covariance from the dense model's earlier layer. Shrinkage estimator with off-diagonal scaling rho in [0,1] (S_rho = D + rho (S - D), D = diag(S)) is exactly the "diagonal target" case.
- Direct evidence [A]: DASH-Q (Kim, Kim, Lee, Seo, arXiv 2604.13806): states that Hessian-based PTQ "degrades at low bit-widths due to noisy curvature estimates from limited calibration data" and proposes "discarding noise-prone dependencies" (diagonal curvature). A search summary adds that diagonal entries stabilise with few calibration samples while off-diagonal entries remain unstable even with many, and that a linear shrinkage family scaling off-diagonals by rho in [0,1] is used [S: search-result summary, not confirmed in the paper text]. Statistically consistent with the RMT picture: diagonal entries are O(1/sqrt(N_eff)) accurate; off-diagonals relative to their size are noisier and there are d^2 of them.
- Recipe A (shrink Hessian toward its diagonal, choose rho by cluster cross-validation): fit H_A on half of sequences, quantize/slice, evaluate layer error on H_B (or output error on held-out data); pick rho from a grid of ~5. Cost: one extra solver run per grid point per layer; for GPTQ that is a multiple of the solve time; can use the cheap surrogate of held-out layer loss tr(E H_B E^T) which needs no model forward. Expected benefit: [I] larger at low bit-width and small calibration sets; at 4-bit with ample data, little.
- Recipe B (Ledoit-Wolf-type analytic rho): closed-form a from S and N_eff (substitute the CLUSTER-level N). Cost O(d^2) extra. [I] Using token N instead of N_eff will under-shrink.

## 5. Heavy tails, outliers, and rotations

### 5.1 Facts about LLM activations
- Massive activations (Sun, Chen, Kolter, Liu, COLM 2024, arXiv 2402.17762): a few scalar entries of the hidden state, in a few fixed feature dimensions, at a few tokens (start token, delimiters), "up to 100,000 times larger" than typical; they act as near-constant biases and trigger attention sinks. [A]
- Outlier features / emergent outlier dimensions (Dettmers et al., LLM.int8(), 2022; Kovaleva et al. 2021 "BERT busters"; Bondarenko et al. 2023) [S]: a small set of hidden dimensions with large magnitude systematically across tokens. They are functionally important: removing them is destructive (LLM.int8 paper, Sun et al. both argue this) [S/A].
- OWL (Yin et al.) discovers layers with more outliers should be pruned less: "sparsity ratio of OWL is proportional to the outlier ratio observed within each layer"; beats Wanda and SparseGPT by 61.22 and 6.80 perplexity at 70% sparsity (abstract) [A: arXiv 2310.05175].

### 5.2 Statistical consequences
- Heavy tails and a few massive tokens make C dominated by < 1% of tokens: C is far from the Gaussian-model assumptions of RMT and shrinkage. Sample covariance error scales with the fourth moment and effective rank (Koltchinskii-Lounici: ||C_hat - C|| ~ ||C|| max(sqrt(r/N), r/N) for sub-Gaussian with r = tr(C)/||C|| the effective rank [S]; for heavy tails with finite fourth moment weaker rates and an extra log factor, Vershynin-style [S]). With massive activations r is tiny in the raw C but the residual spectrum after removing the spike has its own r. [I]
- Consequences: the top eigenvector(s) of C are the massive-activation dimensions (kept by any method), variance-based ordering of the rest is distorted by the fact that sink tokens are 1/T of tokens yet carry orders of magnitude more variance, and cut-point rules assuming homogeneous noise (Gavish-Donoho, parallel analysis) are misled. [I]
- Do NOT simply winsorize outlier dimensions; they carry function. Treat them structurally. Recipe C: split tokens into "sink/massive" (identified by norm > a robust multiple of the median, or by position/BOS and delimiters) and "regular"; always keep the dominant massive-activation directions (a rank-r_0 always-kept subspace, r_0 small); estimate C on regular tokens only (or normalised, below) for the remaining spectrum. Cost: one pass, negligible. [I]
- Robust scatter estimation [S]: Tyler's M-estimator (distribution-free scatter for elliptical data): Sigma = (d/N) sum_i x_i x_i^T / (x_i^T Sigma^-1 x_i), iterate. Ignores the scale of each token (so sink tokens get no extra leverage), needs N > d and is shape-only (scale must be obtained separately). Cost per iteration: O(N d^2) (plus a d x d inverse); roughly 10-50 iterations typical: so ~10-50x the cost of the plain accumulation; subsample tokens to control it. Spatial-sign covariance (x / ||x||)(x/||x||)^T (Locantore et al., Marden, Visuri-Koivunen-Oja 2000): one-pass cost, eigenvectors consistent for elliptical distributions (eigenvalues not) [S]. MCD: breakdown-robust but not computationally feasible for d = 4096 (needs N >> d and combinatorial search) [S]; trimming (drop top-q% by norm or Mahalanobis distance) is the practical form. [S/I]
- Link to RMSNorm [I]: since downstream layers see x/rms(x), the normalised (spatial-sign) covariance is arguably the MORE relevant second-moment object for output error in pre-norm models, and it is what Tyler's estimator targets. This gives a principled reason for token-normalised C (section 3.2), not only a robustness trick.

### 5.3 Rotations as a statistical fix
QuIP/QuIP# (incoherence processing with random orthogonal / Hadamard), QuaRot (Ashkboos et al. 2404.00456), SpinQuant (Liu et al. 2405.16406): multiply W and activations by orthogonal Q so that outliers are spread across coordinates [A].
- QuaRot "rotates LLMs in a way that removes outliers from the hidden state without changing the output", and uses SliceGPT's computational invariance to fuse transformations into weights; Hadamard because exactly orthogonal and applicable in O(d log d) (QuIP#) [A].
- Statistical reading [I]: a random rotation Q makes each coordinate of Q x a sum of many coordinates, so by CLT-type concentration the coordinates are near-Gaussian with roughly equal variance (per-coordinate kurtosis falls, the max/mean ratio falls from O(sqrt d)-ish outlier ratio to O(sqrt(log d))). This turns the heavy-tailed marginal into a light-tailed one, so uniform per-channel quantizers have lower worst-case error and the coordinate-wise Hessian is better conditioned. It is incoherence (max-entry vs RMS) control; it does not change the total variance or the covariance eigenvalues, only the basis. Cost: O(N d log d) for Hadamard at runtime/fused; free for fused rotations.
- Rotation does NOT help SliceGPT directly: it equalises variance, i.e. removes exactly the anisotropy PCA exploits. Rotations and slicing target opposite statistics (flat spectrum vs concentrated spectrum) [I]. DFRot (2412.00648) reports refined rotations that handle massive activations, i.e. random Hadamard performs worse when massive activations are present [A: title/snippet only].

## 6. Subspace stability and uncertainty

### 6.1 Davis-Kahan and gap-free bounds
- Davis-Kahan sin-theta (Davis & Kahan 1970; user-friendly version Yu-Wang-Samworth, Biometrika 2015): for the top-k subspace, ||sin Theta(V_hat, V)||_F <= 2 min( sqrt(k) ||E||_op, ||E||_F ) / (lambda_k - lambda_{k+1}), E = C_hat - C. The eigengap delta = lambda_k - lambda_{k+1} at the cut point controls subspace error. [S]
- LLM spectra are smooth, power-law-like, with no clear gap at an arbitrary cut point, so delta is tiny and the subspace is statistically unstable at k. [I]
- But what we pay is variance/loss, not the angle. Gap-free excess-risk bound [I, derived; assumptions: P, P_hat orthogonal rank-k projectors onto true and sample top-k subspaces]: since P_hat maximises tr(C_hat P), tr(C_hat (P - P_hat)) <= 0, so
  tr(C(P - P_hat)) = tr(C_hat(P - P_hat)) - tr(E(P-P_hat)) <= -tr(E(P-P_hat)) <= ||E||_op ||P - P_hat||_nuc <= 2k ||E||_op.
  Needs no eigengap. Combined with ||E||_op ~ ||C|| sqrt(r/N_eff) (sub-Gaussian) this gives the right scale for how much retained variance is lost to estimation: it is governed by effective rank and N_eff, not by the gap. (A tighter version using the gap gives the (delta/2)||P-P_hat||_F^2 lower bound, so a flat spectrum means the angle can be big at no cost.) The practical message: report stability in terms of retained held-out variance and output loss, not principal angles.

### 6.2 Bootstrap and jackknife over calibration data
- Resampling unit: sequence (or document), never token. Cluster bootstrap [S: Davison & Hinkley; Field & Welsh 2007 for clusters]. Heavy tails: bootstrap is consistent for the eigenvalues/eigenspaces of a sample covariance only with simple eigenvalues and enough moments, and fails for repeated eigenvalues [S: Beran & Srivastava 1985]. At a smooth spectrum, nearly repeated eigenvalues mean bootstrap CIs on individual eigenvectors are unreliable; use bootstrap CIs on functionals (retained variance, loss, k at a target retained fraction) rather than on vectors. [I]
- Cheap implementation [I]: store per-group Gram matrices. Split sequences into G groups (G ~ 10-20), keep C_g (d x d each; for d = 4096, ~67 MB per group in fp32). Delete-one-group jackknife: C_{-g} = C_total - C_g; one eigendecomposition per leave-out, O(d^3) each. Poisson-bootstrap alternative: accumulate B reweighted Gram matrices in a single data pass with Poisson(1) sequence weights. Cost: G or B extra d^3 eigendecompositions per layer and G to B times the d^2 memory; no extra model forward passes.
- Outputs: (a) Var of lambda_j and of tail mass sum_{j>k} lambda_j (jackknife SE); (b) mean ||sin Theta|| between full-data and leave-group-out subspaces; (c) distribution of k_tau = min{k : retained fraction >= tau}; (d) held-out retained variance across groups.
- Empirical ammunition for why this matters: Singh (arXiv 2608.15046) reports that in a controlled GPTQ vs AWQ comparison "changing the calibration draw was sufficient to reverse the observed method ordering" in most test cases, and proposes paired tests, equivalence margins, per-item output release [A]. This is the "uncertainty" piece for evaluation, not for the solution itself.

### 6.3 Choosing the cut dimension from the spectrum, with uncertainty
- Scree/elbow: heuristic, no inference. [S]
- Parallel analysis (Horn 1965): compare eigenvalues with those from column-permuted data; keep those above (e.g.) the 95th percentile of the null. Permute at sequence level and keep structure (otherwise null ignores dependence); only detects non-null structure, not "worth keeping for the loss". [S] Cost: R x (accumulate + eigh) per layer.
- Gavish-Donoho (IEEE-IT 2014): optimal hard threshold for singular values of a low-rank-plus-white-noise matrix: for n x n square matrix, tau = (4/sqrt(3)) sqrt(n) sigma ~ 2.309 sqrt(n) sigma with sigma known, or 2.858 * median singular value if sigma unknown; non-square: omega(beta) ~ 0.56 beta^3 - 0.95 beta^2 + 1.82 beta + 1.43 times the median, beta = aspect ratio [A: arXiv 1305.5870 and search snippet]. Assumes iid noise and a true low-rank signal; use the n_eff aspect ratio. Gives a threshold for "noise-dominated" components, not for a compression budget. [I]
- Minka (NeurIPS 2000) "Automatic choice of dimensionality for PCA": Laplace approximation to the marginal likelihood of probabilistic PCA (Tipping-Bishop 1999) over k; BIC variant. Closed-form from the eigenvalues; assumes Gaussian iid. [S]
- Honest framing [I]: since compression has a hard budget (e.g. 25% parameters), the statistically meaningful question is allocation across layers (section 8) and the uncertainty on loss at the chosen k, not an "intrinsic dimension". These tests are most useful as per-layer relative signals: layers where the noise-dominated component starts earlier can be sliced more.

## 7. Calibration data design

### 7.1 Evidence [A unless noted]
- Williams & Aletras, "On the Impact of Calibration Data in Post-training Quantization and Pruning" (arXiv 2311.09755): impact of calibration data in PTQ and pruning [A, title/snippet].
- "Beware of Calibration Data for Pruning Large Language Models" (Ji et al., arXiv 2410.17711): post-training pruning highly sensitive to calibration data; at sparsity below 50% differences between calibration sets are small, and grow with sparsity: summary reports 0.5% difference at 50% and 2.3% at 60% [S: from search-result summary, check in paper].
- "Is C4 Dataset Optimal for Pruning?" (arXiv 2410.07461) [A, title]; summary in search: perplexity falls rapidly from 8 to 32 samples then slows, suggesting 32-64 samples [S: search summary of one study, task and model dependent; do not generalise].
- "Calibrating Beyond English" (arXiv 2601.18306): performance more sensitive to distributional properties of calibration data than to the number of tokens [A snippet].
- Self-calibration (arXiv 2410.17170), "Think Before You Prune: Selective Self-Generated Calibration" (arXiv 2511.18864), "Preserving LLM Capabilities through Calibration Data Curation" (arXiv 2510.10618), multi-source calibration for high-sparsity pruning (2606.03328): use model-generated or curated or multi-source data [A: titles; headline: calibration choice can change results substantially].
- Singh (2608.15046): calibration draw alone can reverse GPTQ vs AWQ ranking [A].
- SliceGPT itself ran calibration with WikiText-2 and Alpaca [S, from memory of the paper; text not retrieved this session; please verify the exact dataset/size ablation before the lecture].

### 7.2 Statistical framing and recipes [I]
- Domain shift = covariate shift of C. Compression chosen on C_cal and evaluated on C_test; excess variance cost for a subspace Q: tr(Q_perp^T C_test Q_perp) - tr(Q_perp,opt^T C_test Q_perp,opt). If C_test has energy in directions that are small under C_cal, these are deleted.
- Mixture design: keep per-domain Gram matrices C_g (cheap: d^2 each) and form C(w) = sum_g w_g C_g. The weights w are a design parameter: (a) proportional to target traffic (importance weighting), (b) maximin: maximise min_g tr(Q^T C_g Q)/tr(C_g) (a "fair PCA" problem, Samadi, Tantipathananandh, Vempala et al., NeurIPS 2018 [S]) solved by multiplicative-weights over g, each step an eigendecomposition: cost O(T d^3) with T ~ tens of rounds. This is cheap because Grams are precomputed.
- Stratify by sequence source and by position (BOS/early tokens vs later), and by length, so that the sink-token stratum is explicitly controlled (section 5).
- Sample size: choose by the learning curve of held-out layer loss in the cluster bootstrap (section 6.2) and by when relative SE of the quantity of interest drops below the between-method difference you care about; quote N_eff, not token count.
- Report evaluation with multiple calibration draws and paired tests (Singh 2608.15046).

## 8. Per-layer budget allocation as optimisation

### 8.1 Formal problem [I]
Choose compression levels b_l (rank, sparsity, bits, slice fraction) minimising predicted loss sum_l D_l(b_l) subject to cost sum_l c_l(b_l) <= B. D_l from sensitivities (assuming additive layer effects). Lagrangian: minimise D_l(b_l) + mu c_l(b_l) per layer, binary-search mu (convex D_l => exact); if levels are discrete, it is a multiple-choice knapsack: greedy by marginal-cost ratio (convex hull) or DP. Cost: negligible relative to model passes, once D_l(b) curves exist.
- Estimating D_l(b): (a) spectral tail mass, sum of discarded eigenvalues (SliceGPT/SVD-LLM; with whitening this is exactly the layer reconstruction error); (b) Fisher/Gauss-Newton weighted version, 0.5 tr(H_x P_perp C P_perp) (2.4); (c) measured loss increase when compressing that layer alone (one forward per layer per candidate level: expensive, ~ L x levels evaluations; subsample). Additivity is an assumption and wrong (interactions, 3.2); a second-order correction uses pairwise terms (cf. Augmenting Hessians with inter-layer dependencies, 2306.04879 [A, title]).
- Estimation noise propagates: with noisy D_hat_l, the allocation is a plug-in estimator; to avoid winner's-curse overshoot, use jackknife SEs from 6.2 and shrink sensitivities toward a global trend (empirical Bayes) before solving. [I]

### 8.2 Literature (verified at abstract/summary level)
- OWL: non-uniform per-layer sparsity proportional to the layer's outlier ratio (based on activation-weighted weight magnitude outliers); big gains at 70% sparsity (abstract numbers above) [A: 2310.05175]. Follow-ups: DLP "Dynamic Layerwise Pruning" (2505.23807) [A, title], "Beyond One-Size-Fits-All Pruning via Evolutionary Metric Search" (2502.10735) [A, title], "Beyond Layer Importance in Layer-wise Sparsity: an Inter-Layer Perturbation-Absorption Perspective" (2606.15161) [A, snippet: errors are repaired during propagation, layers differ in recovery capability].
- For SliceGPT specifically: "Change Is the Only Constant: Dynamic LLM Slicing based on Layer Redundancy" (Dumitru, Clotan et al., EMNLP 2024). Layer Redundancy (LR) score = cosine similarity between layer input and output; mapped to per-layer slicing percentages. Reports on Llama3-8B at 30% pruning accuracy 49.0% -> 49.8%, perplexity 13.37 -> 12.96; Winogrande 52% -> 57% at 40% pruning; "up to 5% accuracy" and "up to 7% perplexity" in some settings [S: secondary liner.com summary, not the paper; arXiv ID not confirmed].
- ASVD: iterative per-layer sensitivity calibration to choose truncation ranks [A: 2312.05821]. KronQ: new sensitivity metric for mixed-precision allocation from Kronecker-factored curvature [A: 2607.07964]. CoopQ: cooperative-game layerwise mixed precision (2509.15455) [A, title]. HAWQ family (Hessian-trace sensitivity for mixed precision) [S].
- Statistical suggestion [I]: eigenvalue tail mass is a purely second-order-in-activations criterion; a hybrid sensitivity D_l = tail mass weighted by the Fisher trace of the layer's gradient covariance is the natural loss-aware upgrade, and the jackknife SE gives allocation uncertainty.

## 9. Learning the transformation / correcting the deletion

### 9.1 Learning rotations (Stiefel manifold)
- SpinQuant: optimise rotation matrices on the Stiefel manifold with Cayley SGD (Cayley transform of a skew-symmetric matrix keeps orthogonality); finds that "some random rotations lead to much better quantization than others, up to 13 points difference in zero-shot reasoning"; reduces gap to full precision by up to 45.1% relative to QuaRot on LLaMA-3 8B [A: 2405.16406]. Objective: end-to-end loss (cross-entropy) on a small calibration set, weights frozen.
- Cayley: Q(A) = (I - A)(I + A)^{-1} with A skew-symmetric (or the Q_{t+1} = (I - (t/2) A)^{-1}(I + (t/2) A) Q_t update) [S]. Cost per step O(d^3) for a dense d x d rotation (inverse), the reason SpinQuant uses only R1/R2 learned and Hadamard for online rotations [S].
- For slicing [I]: objective min over Q in St(d, k) of L(Q) (end-to-end CE or distillation) or of tr(H_x (I - Q Q^T) C (I - Q Q^T)) (the local quadratic proxy of 2.4). PCA is the closed-form minimiser of the Q-term when H_x = I, so use PCA as initialisation, then a few Riemannian steps on the loss-aware objective. Because the loss near PCA is flat among near-degenerate eigenvalues (6.1), improvements come mostly from rotating within the span of the near-tie eigenvalues and from mixing kept and deleted directions where H_x weights differ. Cost: tens to hundreds of calibration forward-backward passes through the whole model (or blockwise), i.e. between "training-free" and light fine-tuning; no full weight updates. SpinQuant's paper reports how few steps/data suffice but I did not retrieve it [S/I].
- Estimation reading [I]: the number of free parameters is d(d-1)/2 for a full rotation but the data constrain a rank-k Grassmannian, k(d-k) parameters; with small calibration sets this is a highly parametrised M-estimator; regularise by early stopping and by staying near PCA (a penalty tr(I - Q^T Q_PCA...) or low-dimensional parametrisation). 

### 9.2 Bias and mean handling
- Deleting direction q removes q^T x from the stream. If x has a nonzero mean mu (massive activations act as near-constant biases: Sun et al. [A]), then q^T mu is not noise and must be retained or compensated: either include mu in the retained span (use uncentered second moment, which is what PCA of X X^T does, so mean-carrying directions have large eigenvalue and are kept) or add a constant correction b' = W Q_perp Q_perp^T mu to the next layer's bias. [I]
- Centred vs uncentered PCA changes the objective: centred PCA gives the best rank-k approximation of the fluctuations; for the layer-output error with mean included the right matrix is the uncentred second moment. SliceGPT's conversion LayerNorm -> RMSNorm subtracts the mean through the weights [S: from memory of the paper]; check what C they accumulate (centred vs not) in the code. [I]
- Mean-correction recipe (cheap): b' = b + W (I - P) mu_hat; mu_hat from calibration; cost O(m d) per layer. Gain is the removed-mean part of the error; with heavy-tailed sink tokens, estimate mu_hat on regular tokens and handle the sink-token mean separately (otherwise the correction fits the sink token). [I]

### 9.3 Compensation / least-squares correction
- SparseGPT/OBS: update remaining weights so as to minimise layer error given the removed ones (1.1). For quantization, GPTQ error feedback. For low-rank: SVD-LLM "sequential low-rank approximation" update: after truncation, update remaining factors (e.g. fix U, re-solve V by least squares on the whitened problem, then alternate) [A: SVD-LLM abstract mentions "parameter update with sequential low-rank approximation"; details not retrieved].
- General recipe [I]: after compressing layer l, re-solve W' = argmin ||W X_fp - W' X_comp||_F^2 subject to the structure (closed form W' = W X_fp X_comp^T (X_comp X_comp^T + lambda I)^-1 for unconstrained W'; project back to structure). For SliceGPT's orthogonal-slice the unconstrained refit on kept coordinates is allowed (dense kept block) at no inference cost; in-sample gain is from mismatch X_fp vs X_comp (3.1), zero if X_comp = P X_fp exactly in layer 1. Ridge parameter lambda selected by the cluster-CV of 4.3 (this IS a shrinkage estimator, the bias-variance trade-off is explicit).
- Recovery fine-tuning (RFT) as a statistical estimator [I]: with a fixed structure the compressed model has P parameters; fine-tuning on n_FT tokens is an M-estimator of the post-compression optimum; early stopping or LoRA (rank r) provide regularisation. Generalisation concerns: overfitting to the RFT dataset's domain (the same covariate-shift problem as 7); the benefit of RFT diminishes with calibration-set diversity. SliceGPT uses RFT with a small dataset in their paper [S, check]; this is outside "training-free" and should be presented as the boundary of the regime.

## 10. 2025-2026 work framing compression as estimation (found in this session)
- Hessian-estimation focus: DASH-Q (2604.13806) noisy off-diagonal curvature from limited calibration data -> diagonal curvature [A]; GPTAQ (2504.02692, ICML 2025) asymmetric calibration [A]; "Rethinking Residual Errors" (2604.07955, ICLR 2026) compensation-aware error [A]; KronQ (2607.07964) activation x gradient covariances [A]; "Rethinking PTQ: Statistical Pre-Calibration" (2501.09107) [A, title].
- Evaluation statistics: "Certifying Compressed Language Models: An Audit and a Statistical Toolkit" (Singh, 2608.15046): audit of 17 equivalence claims, none with prospective margins or released per-item outputs; churn ~ five times the net accuracy delta; calibration draw reverses method ordering; proposes equivalence margins, paired tests, disagreement metrics [A].
- Theory-flavoured pruning: "Pruning is Optimal for Learning Sparse Features in High-Dimensions" (2406.08658) [A, title] (sample-complexity viewpoint on pruned networks; not about post-training compression of LLMs).
- I did NOT find, in this session, a paper that applies shrinkage/RMT/Davis-Kahan formally to SliceGPT's PCA, nor one that studies bootstrap stability of SliceGPT subspaces. Treat the corresponding recipes as open, [I].

## 11. Recipes and costs (summary table)
| # | Recipe | What it fixes | Extra cost vs baseline pass | Tag |
|---|---|---|---|---|
| 1 | Whitened (C^{1/2}) SVD/rank truncation | PCA/SVD objective vs layer output error | One Cholesky + SVD per matrix, O(d^3) | [A] SVD-LLM, [I] interpretation |
| 2 | Token-normalised or Tyler/spatial-sign C | massive-token leverage; RMSNorm-consistent weighting | spatial sign: none; Tyler: ~10-50x accumulation (subsample) | [S]/[I] |
| 3 | Always-keep massive subspace + C from regular tokens | spectrum distortion, cut-point rules | negligible | [I] |
| 4 | Shrink C or H toward diagonal/pooled target; rho by cluster-CV | noisy off-diagonals; ill-conditioned inverses | O(grid) extra solves, or O(d^2) analytic LW | [A] DASH-Q direction, [I] CV |
| 5 | Sequence-level jackknife/Poisson bootstrap of retained variance, k_tau, angles | no uncertainty on kept subspace | G extra eigh per layer (d^3 each), G x d^2 memory | [I] |
| 6 | Cut dimension by Gavish-Donoho / parallel analysis / Minka, with N_eff | principled "noise-dominated" tail | O(d) after eigh; permutations R x eigh for PA | [S] |
| 7 | Asymmetric (dense-target, compressed-input) calibration | cross-layer error propagation | 2x activation memory, extra d x d cross-moment | [A] GPTAQ / 2604.07955 |
| 8 | Fisher/K-FAC weighting (H_x = E[g g^T]) in the layer objective | gap to true loss | one backward pass; extra d^2 per layer | [A] FWSVD, KronQ; [I] SliceGPT form |
| 9 | Per-layer budget by Lagrangian/knapsack on D_l(b), with jackknife-shrunk sensitivities | uniform allocation | L x levels evaluations if measured; ~0 if analytic | [A] OWL, [S] dynamic slicing, [I] |
| 10 | Domain-Gram mixture or maximin weights | calibration domain shift | T eigh for maximin | [S]/[I] |
| 11 | Learn Q on Stiefel/Grassmannian from PCA init with loss-aware objective | PCA ignores loss | tens-hundreds fwd+bwd passes | [A] SpinQuant, [I] for slicing |
| 12 | Mean-bias correction; dense-target least-squares refit with ridge | deletion bias; propagated error | O(m d) / O(d^3) per layer | [I] |

## 12. Items to verify before the lecture (not checked this session)
1. SliceGPT: whether C is centred; calibration set/size/ablation; whether per-layer Q is computed from sliced or dense activations. (PDF could not be parsed; try arxiv.org/abs/2401.15024 HTML or the repo code.)
2. Exact formulas for BBP eigenvector cosine (Section 4.2) and the Davis-Kahan constants (6.1) against Paul (2007) / Yu-Wang-Samworth (2015).
3. DASH-Q: confirm the linear-shrinkage-with-rho description and the "diagonal stable, off-diagonal unstable" experiment in the full text (currently only via a search summary).
4. "Change Is the Only Constant" arXiv ID and numbers from the paper rather than a secondary summary.
5. Ji et al. 0.5%/2.3% figures.
6. The gap-free excess-variance bound in 6.1 is my derivation; the sign line is written with working. Final statement: tr(C(P - P_hat)) <= 2k ||E||_op.

## Key sources (URLs)
- SVD-LLM https://arxiv.org/abs/2403.07378 ; ASVD https://arxiv.org/abs/2312.05821 ; FWSVD https://arxiv.org/abs/2207.00112
- GPTQ https://arxiv.org/abs/2210.17323 ; SparseGPT https://arxiv.org/abs/2301.00774 ; GPTAQ https://arxiv.org/abs/2504.02692 ; Rethinking residual errors https://arxiv.org/abs/2604.07955
- DASH-Q https://arxiv.org/abs/2604.13806 ; KronQ https://arxiv.org/abs/2607.07964 ; Singh audit https://arxiv.org/abs/2608.15046
- Massive activations https://arxiv.org/abs/2402.17762 ; OWL https://arxiv.org/abs/2310.05175
- QuaRot https://arxiv.org/abs/2404.00456 ; SpinQuant https://arxiv.org/abs/2405.16406 ; SliceGPT https://arxiv.org/abs/2401.15024
- Calibration: https://arxiv.org/abs/2311.09755 ; https://arxiv.org/abs/2410.17711 ; https://arxiv.org/abs/2410.07461 ; https://arxiv.org/abs/2410.17170 ; https://arxiv.org/abs/2511.18864 ; https://arxiv.org/abs/2601.18306
- Gavish-Donoho https://arxiv.org/abs/1305.5870 ; dynamic slicing https://liner.com/review/change-is-only-constant-dynamic-llm-slicing-based-on-layer
- Classical [S, not fetched]: Hassibi & Stork 1993; Ledoit & Wolf 2004; Chen et al. 2010 (OAS); Baik-Ben Arous-Peche 2005; Paul 2007; Benaych-Georges & Nadakuditi 2011; Davis & Kahan 1970; Yu, Wang, Samworth 2015; Tyler 1987; Horn 1965; Minka 2000; Izenman 1975; Koltchinskii & Lounici 2017; Kunstner et al. 2019; Samadi et al. 2018.


---

# SliceGPT baseline (short; read from full arXiv 2401.15024v2 PDF text) 

- Computational invariance: RMSNorm(XQ)Q^T = RMSNorm(X) for orthogonal Q (Eq. 2, App. A.1); Theorem 1 (Eqs. 3-7): W_embd Q, Q^T W_in, W_out Q, Q^T b_out, Q^T W_head. LayerNorm nets are first converted to RMSNorm by absorbing mean-subtraction M into W_out and scale alpha into W_in (Sec 3.2, Fig 3). [V]
- Per-block Q_l (Sec 3.3): residual needs extra Q_{l-1}^T Q_l matrix (D x D, not precomputable). Q_l = eigenvectors of C_l = sum_i X_{l,i}^T X_{l,i} (Eq. 8), sorted by decreasing eigenvalue. [V]
- Slicing (Sec 3.4): delete rows of W_in, columns of W_out and W_embd, rows and columns of the shortcut matrix. [V]
- Eigendecomposition in double precision; FP32 hurts larger models (Table 4, Llama-2 70B 25%: 7.01 vs 4.89 PPL with 128 samples) [V]. Calibration: WikiText-2 or Alpaca; main runs 1024 samples x 2048 tokens; >=128 samples sensible, longer sequences help (App. A.3) [V].
- Results (Table 1, WikiText-2 PPL, 1024x2048 calib): Llama-2 70B dense 3.32, SliceGPT 25% 4.60, SparseGPT 2:4 4.98; Llama-2 7B dense 5.47, 25% 7.24, 2:4 8.69; OPT 66B dense 9.33, 25% 9.68. [V]
- Zero-shot avg w/o RFT (App. Tables 7-8): Llama-2 70B dense 76.57, 25% 69.75 (WikiText calib) / 73.59 (Alpaca calib). With Alpaca RFT (LoRA): 70B 25% 75.62, 30% 74.30; Phi-2 25% 65.24 vs 72.24 dense. [V]  Note abstract says 99/99/90% retention; conclusion says 99/96/87% without RFT, 99/90% for Llama-2 70B/Phi-2 with RFT. [V]
- Speed: Llama-2 70B 25%: 4->3 A100-40GB GPUs, 125 -> 110 ms/token, compute 500 -> 330 GPUms (66%); RTX6000 7->5 GPUs, 1764 -> 1075 GPUms (64%) [V]. Throughput 1.55x at 25% (Table 11) [V]. Matmul-level speedup only ~1.3x at 25% (Tables 12-13) [V].
- Limitations: Llama-2 more sluggish than OPT (flatter spectrum, App. A.4); extra shortcut matrices; small dense models beat pruned-to-same-size ones; OPT RFT didn't work [V].
- Follow-ups/critiques: QuaRot (same authors, uses invariance for Hadamard rotations) [S]; MoDeGPT reports SliceGPT as weak on zero-shot at 25-30% (e.g. Llama-2 7B 30%: avg 48.46%) [V, MoDeGPT appendix]. FASP: SliceGPT ~10x slower, memory-bound beyond 13B on a 4090 [V].
