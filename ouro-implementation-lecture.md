Ouro in Practice.

A Lecture on Implementing and Optimizing a Looped Language Model.

What this is.

Lecture notes covering the implementation program laid out in this folder: the nine-phase execution plan in Ouro execution phase plans for measuring, accelerating, adapting, and compressing the Ouro looped language model, with field context from the companion R D T Guide.

Prerequisites.

Working PyTorch and Hugging Face Transformers knowledge; familiarity with K V caching, LoRA, and post-training quantization at the level of having used them once.

How to read.

Sections 1 - 3 build the conceptual foundation.

Sections 4 - 12 walk the nine phases in execution order - each one answers why this phase exists, what the core mechanism is, and how you know it worked.

Section 13 extracts the engineering lessons that repeat across phases.

Section 14 is a self-test.

1.

Motivation: why loop a transformer at all?.

A standard transformer runs each of its layers exactly once per forward pass.

A looped (recurrent-depth) transformer takes a smaller stack of layers and runs it k times, feeding the output hidden state back in as input.

The weights are shared across iterations; the computation is not.

This buys one unconditional win and creates two conditional costs.

Keep this ledger in your head for the entire lecture - every phase in the implementation plan is an entry in it.

Here is the same table in spoken form.

First: Ledger item: Weight memory.

Effect of looping k times: L unique layers do the work of a k times L layer stack so you store one over k of the weights.

Status: Always a win.

This is the reason looped models fit on hardware their dense-quality peers don't.

Second: Ledger item: K V cache.

Effect of looping k times: Naively, each loop iteration writes its own keys and values, creating up to k times decode cache.

Status: Conditional cost.

Phases 3 and 8 exist to erase it.

Third: Ledger item: Latency.

Effect of looping k times: k loops = k sequential passes per generated token.

Status: Always paid, unless you exploit early exit, drafting, or routing.

Phases 1, 2, 4, and 7 exist to reduce it.

Why would running the same weights repeatedly add quality at all? The intuition, supported by the mechanistic literature the guide surveys, is that the loop performs iterative refinement: each pass moves the hidden state closer to a fixed point that encodes the answer.

Easy inputs converge in one or two passes; hard inputs need all of them.

That variability is not a nuisance - it is the entire optimization surface this implementation plan exploits.

Key idea number one.

A looped model turns "how much compute does this input get?" into a runtime knob.

Everything in this folder is about measuring that knob, then monetizing it.

2.

Meet the model: Ouro's control surface.

Ouro (ByteDance, Apache-2.0) is the flagship looped checkpoint family and the target of the entire plan: Checkpoints: ByteDance Ouro-1.4 billion and ByteDance Ouro-2.6 billion, plus -Thinking variants (reasoning S F T on the base models).

Training: 7.7 trillion tokens, trained at 4 recurrent steps (you will see this written as R4 or T equals 4).

Headline claim: the 2.6 billion looped model rivals about 8 billion dense models on reasoning - large-model quality at small-model weight memory, which is exactly the ledger from Section 1 paying out.

The plan targets Ouro-1.4 billion for development and 2.6 billion for confirmation runs, with the Thinking variants deferred until the harness works.

This ordering is deliberate: iterate where each experiment is cheap, confirm where it matters.

2.1 The two knobs.

Ouro exposes its loop behavior through two config fields: The two control fields are config dot total U T steps, which sets the recurrent depth, and config dot early exit threshold, where one point zero means always run the full configured depth.

ut stands for universal transformer steps.

These two fields are the control surface every phase manipulates.

2.2 The three environment landmines.

The plan treats these as first-class engineering facts, and so should you: 1.

Pin transformers equals 4.54.1.

The model ships custom code (trust remote code equals true), and the model card warns that transformers greater than or equal to 4.56.0 breaks it.

A community cache fix was merged upstream, but the pin remains the documented safe path.

2.

Set loop config before from pretrained. The loop count is baked into the model at load time.

Editing config.total U T steps after loading silently does nothing - one of the most common failure modes in Phase 0's table.

3.

v L L M or S G Lang always run full depth.

Official serving integrations exist, but they do not support Ouro's adaptive early exit - they execute all total U T steps every time.

Any experiment that depends on average-case early exit must use the Hugging Face path.

This single engine limitation motivates the entire router design in Phase 7.

2.3 The extrapolation ceiling.

You might hope to "think harder" at inference by setting T equals 8 on a model trained at T equals 4.

Empirically this fails: base Ouro peaks at its trained depth (T equals 4) and degrades at T equals 5 - 8, and the Thinking variants are near-useless at T equals 1, peaking around T equals 3 - 5.

Independent scaling analyses (Parcae) confirm test-time looping follows a saturating curve whose ceiling sits near the mean training recurrence.

Key idea number two.

You cannot loop your way past what training established.

The realistic optimization direction is downward - finding inputs that need fewer than 4 loops - not upward.

3.

The implementation philosophy.

The phase plans compress their strategy into one sentence, which is worth memorizing because it explains the ordering of everything that follows: First measure loop behavior, then reduce memory or latency, then train small loop-aware adapters, then compress with loop-aware calibration.

Measurement (Phases 0 - 2) comes before optimization (3 - 4), which comes before training (5), which comes before compression (6), which comes before productization (7).

The stretch goal (8) branches off after Phase 3.

The dependency order is phase zero for environment, phase one for loop sweeps, phase two for trajectory diagnostics, then phase three for cache policy and phase four for self-speculative decoding.

From there, phase five handles per-loop LoRA, phase six handles loop-aware quantization, phase seven handles routing and serving, and phase eight is the MELT-lite stretch goal.

The README is blunt about why the ordering is non-negotiable: without loop-depth and trajectory baselines, you will not know whether a training or compression change improved reasoning or merely changed surface likelihood.

3.1 The global acceptance rule.

Every phase must produce five artifacts before you move on: 1.

A reproducible script or config.

2.

A machine-readable output file (JSONL or Parquet).

3.

A human-readable summary Markdown.

4.

A test that prevents the same bug from silently recurring.

5.

A regression comparison against the Phase zero or Phase one baseline.

Item 4 is the one teams skip and regret.

Notice as we go how each phase converts its scariest failure mode into a permanent unit test.

3.2 The repository as a map of the plan.

The proposed repo layout mirrors the phase structure - each ouro extension module is one phase's mechanism: The proposed repository mirrors the plan.

The model loader belongs to phase zero, instrumentation to phase two, cache policies to phase three, speculative decoding to phase four, per-loop LoRA to phase five, quantization to phase six, and routing to phase seven.

The evals and tests folders are the permanent measurement harness.

3.3 Measurement discipline.

Two global standards apply to every experiment: Log everything that could explain a difference later: model revision, transformers or torch or CUDA versions, G P U name, dtype, total U T steps, early exit threshold, seed, git commit, plus per-run latency (TTFT, inter-token mean or p fifty and p ninety five), tokens per second, and peak V RAM.

Evaluate on a small, fixed suite built for loop sensitivity, not leaderboard breadth: short factual QA (should not need loops), G S M eight K-style arithmetic and MATH five hundred-style problems (loop-depth sensitive), synthetic multi-hop facts, small code tasks (regression canary), and long-context retrieval snippets (cache canary).

Iterate on the small suite; lock a larger one only once the harness is stable, and never tune on the locked suite.

3.4 Hardware: running the plan on Shadeform.

Every phase is single-G P U, and each phase file now pins the cheapest Shadeform marketplace tier that fits its peak V RAM: an R T X fifty ninety (32 gigabytes card, plan to about 29 gigabytes usable) by default, escalating to an RTX A six thousand (48 gigabytes card, about 46 gigabytes usable) only where measured need demands it.

The full assignment table and operational checklist live in the phase-plan README; the shape of the result is worth internalizing: Here is the same table in spoken form.

First: Work: Phases 0 - 4, 6, 7 - all measurement, inference, cache, self-spec, quantization, and routing work.

Instance: R T X fifty ninety.

Why: Peaks top out near 18 gigabytes.

Weights are small (1.4 billion about 2.8 gigabytes, 2.6 billion about 5.3 gigabytes B F sixteen); even two resident copies (Phase 4) or three replicas (Phase 7) fit.

Caches and traces dominate and are controllable.

Second: Work: Phases 5 and 8 on Ouro-1.4 billion.

Instance: R T X fifty ninety.

Why: 4-loop BPTT plus large-vocab F P thirty-two loss spikes land at about 12 - 20 gigabytes with the fit rules in each phase file: gradient checkpointing, micro-batch 1, shared teacher or student weights, capped attention alignment.

Third: Work: Phases 5 and 8 on Ouro-2.6 billion; full-resolution attention alignment; quantization-kernel fallback.

Instance: A six thousand.

Why: Training-loss logit spikes and the O of sequence length squared alignment term can exceed 32 gigabytes, and Blackwell (sm120) wheels for G P T Q or A W Q toolchains may lag - the A six thousand's Ampere sm86 is universally supported.

Two disciplines carry over from the rest of the lecture.

First, the assignments are estimates until measured: the harness logs peak V RAM in gigabytes on every run, and Phase zero or Phase one numbers should overwrite the table.

Second, latency results are G P U specific - a Pareto frontier (Section 5) or speculative-decoding speedup (Section 8) measured on the 5090 must never be merged with A six thousand rows; G P U name sits in the logging schema precisely so this mistake is catchable.

One operational habit matters more than any number here: Shadeform instances are ephemeral and billed hourly, so sync outputs folder off-instance after every step and tear the instance down when idle - the Phase 0 snapshot manifest is what makes re-provisioning cheap.

4.

Phase 0 - Reproducibility before experimentation.

Source file: phase zero, environment and baseline.

Risk: Low 4.1 Why this phase exists.

Ouro is custom remote code with version-sensitive behavior.

The failure mode Phase 0 prevents is subtle and expensive: a later "speedup" that is actually broken generation, an incompatible library, or an untracked patch.

Every result in Phases 1 - 8 is only as trustworthy as this baseline.

4.2 Snapshot the model source like it's your own code.

Because the modeling files arrive via trust remote code equals true, the plan's first move is to make them inspectable and diffable: 1.

snapshot download the full repo locally.

2.

Write a manifest with SHA-256 hashes of every file (model snapshot manifest dot json).

3.

Patch only local copies, keeping a reproducible diff against the original snapshot.

This turns "mystery remote code" into a pinned, auditable dependency.

When Phase 2 patches the recurrent loop and Phase 3 patches the cache path, the diff against this snapshot is the record of exactly what changed.

4.3 The config-before-load pattern.

The loader (ouro extension model loader) encodes landmine number two from Section 2.2 structurally, so nobody can get it wrong: The two control fields are config dot total U T steps, which sets the recurrent depth, and config dot early exit threshold, where one point zero means always run the full configured depth.

Every later phase loads the model through this one function.

Centralizing the pattern is what makes "loop count is a config knob" actually true in practice.

4.4 Smoke tests and the acceptance gate.

Run greedy generation on a fixed math prompt at T equals 1, T equals 2, and T equals 4, logging elapsed time, generated tokens, and peak V RAM for each.

You proceed only when: all three depths load and generate, V RAM or time are logged, the snapshot manifest exists, the unit tests pass (model loads at T equals 1; config exposes both knobs), and you have known-good outputs for the same prompt at multiple depths.

That last item seems trivial but is the first scientific artifact: three depths, one prompt, visible behavioral difference - the null hypothesis everything later is compared against.

5.

Phase 1 - Mapping the loop-depth Pareto frontier.

Source file: phase one, loop depth benchmarking.

Risk: Low 5.1 The questions.

Before modifying anything, establish what the knob is worth in its factory state: Which tasks actually benefit from loops, and which tolerate shallow ones? What does each extra loop cost in latency and V RAM? Is adaptive exit (via the Hugging Face path) worth it versus fixed depth? Do base and Thinking variants behave differently? (Yes - evaluate them separately; a Thinking model collapsing at T equals 1 is expected, not a bug.) 5.2 Design.

Sweep fixed depths T in one through six on both model sizes with deterministic greedy decoding (do sample equals false), warmup generations before timing, and repeats for latency stability.

Note the sweep deliberately includes T equals 5 and T equals 6 - expected to be useless per Section 2.3 - because confirming the ceiling on your tasks is cheap and definitive.

Then sweep early exit threshold values of one point zero, zero point eight, zero point six, zero point four, and zero point two at T equals 4 on the Hugging Face path (threshold one point zero means full depth baseline).

One row per example per setting, flushed incrementally so a crash can't erase completed work.

Each row carries quality (exact match for math, regex-normalized answers for QA, pass or fail for code), runtime (TTFT, ITL percentiles, tokens per second, peak V RAM), and the realized loop count when the model exposes it - configured and realized depth can differ under adaptive exit.

5.3 Pareto analysis and the decision document.

A setting is Pareto-optimal if no other setting has both higher-or-equal quality and lower-or-equal latency with at least one strict improvement.

The phase's real deliverable is not the parquet file - it's a decision memo: recommended T per task bucket, best exit threshold, average realized loops, and an explicit "do not use" list (settings that regress past tolerance, or that "win" latency only through truncated or broken generation - always check the cutoff rate before believing a speedup).

The tolerances are written down before results arrive, e.g.

product-like QA accepts less than or equal to 0.5 - 1 point quality loss for greater than or equal to 10% latency reduction.

Pre-committing to thresholds is what keeps the analysis honest.

Key idea number three.

Phase 1 converts "Ouro has a loop knob" into "for this task, T equals 2 is free money; for that task, never go below 4." Every subsequent phase spends from this map.

6.

Phase 2 - Opening the loop: trajectory diagnostics.

Source file: phase two, loop trajectory diagnostics.

Risk: Low-Medium 6.1 From black box to dynamical system.

Phase 1 told you what depth does to accuracy.

Phase 2 instruments why, by recording the model's internal trajectory across loop iterations.

For each prompt and each loop index t: Here are the entries in spoken form.

First: Metric: Logit entropy.

Definition: H(p sub t).

What it tells you: Is the model getting more confident per loop?.

Second: Metric: Top-1 token & margin.

Definition: argmax, p one minus p two.

What it tells you: Is the answer readout stable?.

Third: Metric: K L and J S divergence to next loop.

Definition: K L divergence from p at loop t to p at loop t plus one.

What it tells you: Is the prediction distribution converging?.

Fourth: Metric: Hidden cosine.

Definition: cosine between h at loop t and h at loop t plus one.

What it tells you: Is the latent state converging?.

Fifth: Metric: Relative update norm.

Definition: the norm of the current hidden update divided by the previous hidden norm.

What it tells you: Are updates shrinking (converging) or exploding?.

Sixth: Metric: Answer-at-loop.

Definition: parsed answer per t.

What it tells you: Would exiting at loop t have been correct?.

Start with the final-answer-token logits only; expand to all generated tokens later.

Compute divergences in float32 via log softmax - never clip probabilities inside the model path itself.

6.2 Two instrumentation strategies.

Strategy A (preferred): patch the local model snapshot (from Phase 0) so the recurrent loop optionally returns loop logits and loop hidden lists, gated by a return loop diagnostics flag.

One forward pass yields the whole trajectory.

Strategy B (fallback): run the same prompt at total U T steps equal to one, two, three, and four and treat the four final outputs as trajectory proxies.

Easy and patch-free, but about 4 times the compute and not guaranteed identical to in-loop states if gates or caches behave differently.

Explicitly a temporary baseline.

The non-negotiable safety rail for Strategy A: with diagnostics disabled, logits must match the unpatched model to less than one E minus five.

This "no-op equivalence" test is the pattern to internalize - every observation tool must prove it doesn't perturb the system when off, and the proof lives in the test suite forever.

6.3 Early-exit rules, and the wrong-attractor trap.

From the trajectory metrics, five candidate exit rules are evaluated offline (no serving risk - just replay the recorded trajectories under each rule): 1.

K L between consecutive loops below epsilon for m consecutive loops.

2.

Parsed answer unchanged for m consecutive loops.

3.

Entropy plateau and top one margin above threshold.

4.

Hidden update norm shrinkage below epsilon.

5.

Conservative combination: K L small AND answer stable AND update small - the recommended first production candidate.

Why insist on combining signals? Because of the most instructive failure mode in the phase's table: K L tiny but answer wrong - the model has converged to a wrong attractor.

Convergence and correctness are different properties; a stability-only exit rule will confidently early-exit on exactly the examples where extra loops were needed.

Related: intermediate readouts can look healthy while the final answer degrades (the literature calls this the readout blind spot), so always validate exit rules against the final task metric, not per-loop proxies.

6.4 Phase 2 is the load-bearing phase.

Its outputs feed almost everything downstream - this is why the plan forbids starting Phases 5, 6, or 8 without it: Phase 3 uses trajectories to verify cache reuse doesn't distort loop states.

Phase 4 uses loop stability to choose the draft depth.

Phase 5 uses trajectory failures (examples where shallow loops flip the answer) as training signal.

Phase 6 turns trajectory fidelity into its quantization safety metric.

Phase 7 uses trajectory features as router inputs.

7.

Phase 3 - K V cache: the prefill or decode asymmetry.

Source file: phase three, K V cache optimization.

Risk: Medium 7.1 The problem and the empirical anchor.

Naive looped decoding stores a K V cache per loop: at T equals 4, roughly 4 times the loop-related decode cache.

The guide's empirical anchor (Ouro-1.4 billion) says most of that is waste: Here are the entries in spoken form.

First: Decode-time policy: Full per-loop cache (baseline).

G S M eight K: 78.92.

MATH five hundred: 82.40.

Verdict: reference.

Second: Decode-time policy: Reuse final loop's K V only.

G S M eight K: 78.85.

MATH five hundred: 80.40.

Verdict: near-lossless, 4 times decode-cache cut.

Third: Decode-time policy: Reuse averaged K V.

G S M eight K: works.

MATH five hundred: works.

Verdict: viable ablation.

Fourth: Decode-time policy: Reuse first loop's K V.

G S M eight K: about 18.7.

MATH five hundred: -.

Verdict: collapse.

Fifth: Decode-time policy: Any reuse during prefill.

G S M eight K: -.

MATH five hundred: -.

Verdict: costs more than 10 points - never.

Hence the implementation rule, verbatim from the plan: In plain terms: Prefill: keep full per-loop cache.; Decode: reuse only the final loop K V cache by default.

Why the asymmetry? During prefill the model is still building its representation of the prompt - intermediate loops genuinely read intermediate-loop context, and collapsing them corrupts the refinement process itself.

By decode time, the prompt representation has converged; the final loop's K V is the converged reading, so new tokens attending only to it lose almost nothing.

And first-loop reuse fails for the mirror-image reason: loop 1's K V is a rough draft that later loops were supposed to refine.

7.2 Implementation shape.

A CachePolicy enum: fullloopcache, finalloopdecode (target), averageloopdecode (ablation), first-loop decode (negative control - never ship), no cache (debugging).

Inspect pastkeyvalues before assuming anything - remote-code cache objects need not follow Hugging Face's standard tuple format.

Print the structure after prefill and after one decode step first.

If Hugging Face generate() hides the cache handoff, write a minimal explicit greedy loop: prefill with full policy to transform cache to feed one token at a time.

Owning the loop makes the prefill or decode boundary impossible to blur - and a test (testprefillusesfullcacheevenwhenpolicyfinal loop) pins that boundary permanently.

7.3 Detecting silent damage: logit drift.

Exact-match accuracy is too coarse to catch early corruption, so the phase compares full-cache versus policy runs token by token: max or mean absolute logit difference, K L, and top one agreement (target greater than or equal to 95% on deterministic short generations).

A divergence spike at the first decode token is the signature failure - it means the cache transform fired during prefill or selected the wrong loop.

Acceptance: memory reduction visible and monotonic with generation length (if memory doesn't fall, the transform likely still holds references to the old loop tensors - a pure Python-object-graph bug that quality metrics will never reveal), quality within the Phase 1 tolerance, and no drift spikes.

Notice the epistemic role of first-loop decode: it's kept because it's known-bad.

If your harness doesn't show first-loop reuse collapsing, your harness is broken - a built-in positive control for the measurement apparatus itself.

8.

Phase 4 - Self-speculative decoding: the loop as its own draft model.

Source file: phase four, self-speculative decoding.

Risk: Medium 8.1 The idea.

Classical speculative decoding needs a separate small draft model.

A looped model carries its own draft family inside: the same weights at lower loop count.

The self-speculative flow is: draft cheaply with T equals one or T equals two, verify with T equals four, and accept the longest prefix that exactly matches the verifier.

The greedy algorithm: draft block size tokens cheaply; run the T equals 4 verifier once over prompt plus block; accept the longest prefix where the verifier's greedy choices match the draft; if nothing matches, emit the verifier's first token (guaranteeing greater than or equal to 1 token of progress per iteration); repeat.

8.2 The exactness contract.

The phase's central discipline is a provable correctness property: In plain terms: self-spec text must equal the full T equals four greedy text (exactly, token for token).

Under greedy decoding, strict prefix-verification must reproduce the full-depth output exactly - every emitted token is either verified as the T equals 4 greedy choice or produced directly by the verifier.

If outputs differ, you don't have a quality trade-off; you have a bug.

Only after exact mode works may you explore approximate acceptance rules for extra speed - and then results from the two modes must never be mixed in the same chart.

8.3 An engineering ladder, not a single build.

The plan sequences three implementations by risk: Option A - two model instances (one at T equals 2, one at T equals 4): trivially correct, doubles weight V RAM.

Correctness prototype only.

Option B - one model with mutable per-forward loop count: requires understanding Ouro's custom forward path; the production direction.

Option C - shared weights under two lightweight wrappers: most engineering, best serving shape.

Prototype where correctness is easy to establish, then migrate the validated behavior toward efficiency - with the exactness test guarding every migration step.

8.4 What decides success.

Speedup here is not free; it's an economics question measured by acceptance rate, mean accepted prefix length, and verifier-call count.

Two failure modes matter most: acceptance about 0 (draft too shallow for the task - use T equals 2 or smaller blocks; Phase 2 already told you which tasks stabilize early) and negative speedup (verification overhead exceeds drafting savings).

Hard math may simply route to full depth - which is fine, because Phase 7 will formalize exactly that.

After exact-mode acceptance, re-run the ablation with Phase 3's finalloopdecode policy - the wins compound.

A stretch probe for Efficient-Parallel-Samplers-style decoding is explicitly fenced off as a research branch: prototype tiny, and stop if correctness demands architecture surgery deeper than cache or loop wrappers.

9.

Phase 5 - Per-loop LoRA: breaking weight-tying symmetry cheaply.

Source file: phase five, per-loop LoRA S F T.

Risk: Medium-High 9.1 Why loops might want to differ.

Weight sharing is the source of the memory win, but it imposes a symmetry: loop 1 (rough encoding) and loop 4 (final refinement) plausibly want different computation, yet must use identical weights.

The literature's standard fix: keep the shared backbone frozen and attach small loop-indexed LoRA deltas - loop i gets its own low-rank adapter, so each pass can specialize for a few megabytes instead of unsharing gigabytes.

The per-loop LoRA wrapper chooses an adapter by loop index, applies that low-rank delta to the shared linear layer, and leaves the frozen base projection intact.

Practical starting point: rank 8, alpha equals sixteen, on query projection, output projection, up projection, and down projection; raise rank or retarget modules only if the adapter underfits.

The wiring challenge is real but shallow: the loop index must be plumbed into the adapted linears - exactly the loop-context plumbing Phase 2's instrumentation already built.

9.2 The zero-init gate.

Because B matrices start at zero, the wrapped model must be bitwise-boring before training: At zero initialization, the base logits and LoRA logits must match to less than one E minus five before training starts.

If this fails, do not train.

A wiring mistake that shifts outputs at zero-init would otherwise be laundered into the trained weights, and every downstream eval would measure the bug.

Same pattern as Phase 2's no-op test: prove the intervention is invisible when off.

9.3 Controlled experiment design.

Five variants, each isolating one question - is any tuning enough (B: shared LoRA)? does loop-indexing add anything over it (C)? does sampling T sampled from two, three, and four per sequence during training buy depth robustness (D)? does distilling from the frozen T equals 4 teacher rescue shallow-loop quality (E: cross-entropy plus a beta-weighted K L distillation term from the T equals four teacher)? Variant E is the strategically interesting one: it doesn't chase peak quality - it purpose-builds a better cheap model at T equals 2 or T equals 3, which is precisely what Phase 4's drafter and Phase 7's router want to buy.

An optional stability regularizer penalizes exploding update norms - gently; the goal is preventing explosive loop dynamics, not making loops identical (identical loops would defeat the point of depth).

9.4 Evaluate off the training point.

Everything is evaluated at T equals one through six, never just the trained depth, with three named deltas: low-loop recovery (score at T equals two minus frozen score at T equals two), full-depth preservation (score at T equals four minus frozen score at T equals four), and over-loop robustness (T equals five and T equals six).

The rejection criteria bite: reject if full-depth quality drops on the locked suite, if the adapter only helps the training distribution, or - importantly - if per-loop LoRA merely ties shared LoRA while adding complexity.

Complexity must pay rent.

10.

Phase 6 - Loop-aware quantization: error compounds along the trajectory.

Source file: phase six, loop-aware quantization.

Risk: Medium-High 10.1 Why dense-model PTQ recipes mislead here.

Three independent 2026 results (LoopQ; Hyperloop's loop-aware G P T Q; the ML Collective edge study) converge on one message: in a looped model, quantization error is recursive - the same quantized layer's error feeds back into its own next-iteration input.

Three loop-specific failure modes follow: (1) one shared layer sees different activation distributions at loop 1 versus loop 4, so single-role calibration fits none of them; (2) state reuse across loop transitions propagates error; (3) errors accumulate along the trajectory.

Baseline W four A four doesn't degrade gracefully - it collapses (greater than 200 perplexity on LAMBADA, with spikes at loop transitions).

The edge study adds the most deceptive finding: aggressive compression can preserve local token predictions while destroying global reasoning - per-token accuracy holds while exact-solution accuracy collapses to zero.

Hence the phase's evaluation dogma: exact match on end tasks, never perplexity alone.

Perplexity is a local metric; looped reasoning is a global property.

10.2 The one-sentence fix, and the safety metric.

Never calibrate a shared layer as if it were used once.

Collect activation statistics per loop index, then calibrate each shared layer on the union across all its loop roles - Hyperloop's version for G P T Q: aggregate the Hessian estimate over all iterations of the layer (their reference recipe: INT4, 1024 calibration sequences, group size 128).

Per-loop stats also serve as a risk map: layers whose loop one versus loop four distributions differ most are flagged as fragile before any quantization runs.

The safety metric generalizes Phase 2's machinery - trajectory fidelity: Trajectory fidelity is the cosine similarity between the full-precision hidden state and the quantized hidden state, measured by layer and by loop index.

aggregated by layer, by loop, and - most diagnostic of all - across loop transitions (t to t plus one), because that's where recursive error spikes first.

Safety thresholds are pre-committed: quality drop less than or equal to 1 point, mean fidelity greater than or equal to 0.98 with no catastrophic layer or loop dips, top one agreement greater than or equal to 95%.

10.3 The ladder and the rescue playbook.

Compression proceeds in strictly increasing aggressiveness - do not start at W four A four: Here are the entries in spoken form.

First: Stage: Q0.

Format: B F sixteen.

Role: baseline.

Second: Stage: Q1.

Format: W eight A sixteen.

Role: sanity compression.

Third: Stage: Q2.

Format: W four A sixteen, standard calibration.

Role: negative or standard control.

Fourth: Stage: Q3.

Format: W four A sixteen, loop-aware calibration.

Role: first real target.

Fifth: Stage: Q4.

Format: Q three plus per-layer rescue.

Role: safer target.

Sixth: Stage: Q5.

Format: W four A four loop-aware.

Role: stretch only.

Q2 plays the same role as first-loop decode in Phase 3: the expected-worse control that proves the loop-aware treatment (Q3) actually causes the improvement.

For layers that stay fragile, the rescue ladder is graded - keep B F sixteen, drop to W8, shrink group size, per-channel scales, loop-aware activation scaling - and the plan explicitly permits the first accepted checkpoint to be mixed precision.

A uniformly-INT4 model that fails reasoning is worth less than a mixed model that works.

One caveat carried from the guide: compression (smaller checkpoint, less V RAM) and latency are separate wins - without hardware-suited kernels, INT4 can run no faster than B F sixteen.

11.

Phase 7 - The hardness router: adaptive compute without engine support.

Source file: phase seven, hardness router and serving.

Risk: Medium 11.1 Reframing early exit as a systems problem.

Recall landmine number three: production engines run Ouro at fixed full depth - the model's internal adaptive exit is unusable there.

Phase 7's move is to hoist the depth decision out of the model and into the serving layer: a request-level router picks, per prompt, In plain terms: T star is the smallest loop count whose answer matches the full-depth answer (or the target).

Labels come free from Phase 1's sweep (every example already has outcomes at every T - when no ground truth exists, T equals 4 serves as pseudo-teacher, flagged as a weaker label).

Request-level routing is chosen deliberately over token-level: it needs no model surgery, works with any engine, and is auditable.

Token-level adaptivity is the literature's frontier, not a first implementation.

11.2 Features and models: buy signal only when needed.

Three feature tiers, in increasing cost: A - prompt-only (length, math-symbol density, code-likeness, question type; zero model cost), B - one-loop probe (run T equals 1, read entropy or top one margin or answer-parse confidence - a hardness measurement, since Phase 2 showed easy prompts converge almost immediately), C - two-loop stability (K L between T equals 1 and T equals 2 outputs, answer stability; strongest and priciest).

Start with A plus B.

The same escalation logic governs the classifier: rules to logistic regression or GBDT to small MLP only if tabular fails.

The recommended first router is prompt-only gradient-boosted trees plus a conservative fallback to T equals 4 whenever confidence less than 0.75.

11.3 Asymmetric loss - the ethical core of the router.

The two error types are not symmetric: undercompute that flips a correct answer to wrong is strictly worse than overcompute that wastes loops.

The config makes the asymmetry explicit - loss weight undercorrect: one point zero versus loss weight overcompute: 0.25 - and the eval tracks undercorrect rate (router chose cheap-and-wrong where T equals 4 was right) as a first-class metric, with the acceptance gate requiring those cases be manually reviewed, not just counted.

Acceptance targets: quality within less than or equal to 0.5 - 1 point of always-T equals 4 while cutting average loop count greater than or equal to 15%.

11.4 Serving topologies.

Three deployment shapes, exploiting that fixed-depth Ouro is engine-supported: 1.

Hugging Face dynamic server - one process, per-request depth; flexible, slower per token.

2.

Fixed-depth replicas - ouro-t2-server, ouro-t3-server, ouro-t4-server behind the router; each replica is a plain fixed-depth model, so v L L M or S G Lang work fine.

The engine limitation is routed around, not waited on.

3.

Hybrid - v L L M replicas for easy high-throughput traffic, Hugging Face path for uncertain or diagnostic traffic.

Production logging closes the loop (selected depth, confidence, fallback flag, latency, answer hash - with prompt hashes rather than raw text unless privacy policy allows), turning every served request into future router training data.

12.

Phase 8 - MELT-lite: one gated cache (stretch goal).

Source file: phase eight, MELT-lite single-cache stretch.

Risk: High 12.1 From selection to fusion.

Phase 3 selected one loop's K V cache, a zero-training inference trick).

MELT (the paper this phase miniaturizes) changes the model's memory design: a single K V cache per layer, shared across all loops, updated each iteration by a learned gate - constant memory in reasoning depth.

That is a change to recurrent dynamics, which is why it needs training and sits at the end of the dependency graph, gated on Phase 3's cache understanding.

MELT-lite strips the idea to its cheapest testable form: In plain terms: The new shared K V cache equals alpha times the old shared K V cache, plus one minus alpha times the current loop K V cache.

with alpha a learned scalar per (layer, loop) - sigmoid-parameterized, initialized near 0.8 (biased toward retaining memory).

Vector (per-channel) gates only if scalars demonstrably underfit.

12.2 Minimal-intervention training.

Three stages, each adding trainable surface only if the previous underfits - with the base model frozen throughout: 1.

Gate-only distillation: train just the alpha parameters (dozens of them!) to match the full-cache teacher: K L from student to teacher logits, plus lambda times hidden-state mean squared error.

This is a pure hypothesis test: is cache interpolation enough? 2.

Then add tiny LoRA on attention output projections - reusing Phase 5's machinery - if gates alone can't match the teacher.

3.

Then add task cross-entropy loss on final answers, only once distillation is stable.

The unit tests are boundary-condition proofs: alpha equals one must reproduce pure old-cache behavior, alpha equals zero pure current-cache; the gated cache must remain structurally valid for the next decode step.

And the trained gate values are themselves an interpretability artifact - alpha near 1 in some layer says "long-term memory layer"; alpha near 0 says "recompute every loop." 12.3 Kill criteria as a feature.

The phase pre-declares when to stop: gate-only training can't match teacher logits; memory reduction doesn't beat Phase 3's final-loop reuse in your actual serving mode; or quality losses concentrate on exactly the reasoning tasks Ouro exists to be good at.

A stretch goal with explicit kill criteria is research; one without them is a sunk-cost trap.

13.

The through-lines: ten engineering lessons.

Zoom out and the nine phases repeat a small set of moves.

These generalize far beyond Ouro: 1.

Measure before modifying.

Three full phases of measurement precede the first optimization.

The plan's own words: without baselines you can't distinguish "improved reasoning" from "changed surface likelihood." 2.

Reproducibility is an artifact, not a habit.

Pinned versions, SHA-256 manifests, logged revisions or seeds or commits - all machine-checkable, none aspirational.

3.

Every intervention ships an equivalence proof.

Diagnostics off implies logits match (Phase 2).

LoRA at zero-init implies logits match (Phase 5).

Full-cache policy implies identity (Phase 3).

Greedy self-spec implies exact T equals 4 output (Phase 4).

B F sixteen versus B F sixteen fidelity implies about 1.0 (Phase 6).

alpha at zero or one implies pure endpoints (Phase 8).

4.

Carry negative and positive controls.

first-loop decode should collapse; standard G P T Q should underperform loop-aware.

If your controls don't behave, distrust the harness before the treatment.

5.

Local metrics lie about global properties.

Perplexity and per-token agreement survive compressions that zero out exact-match reasoning.

Evaluate the property you actually care about.

6.

Loop transitions are where things break.

Cache handoffs, quantization error spikes, drift at the first decode token - instrument boundaries hardest.

7.

Exploit asymmetries.

Prefill is not the same as decode (Phase 3).

Undercorrect is not the same as overcompute (Phase 7).

Draft is not the same as verify (Phase 4).

Uniform treatment of asymmetric situations leaves value on the table.

8.

Convergence is not the same as correctness.

A stable trajectory can be stably wrong.

Exit rules and routers must consult answer-level signals, not just dynamics.

9.

Escalate complexity only against demonstrated need.

Rank 8 before 16; scalar gates before vector; rules before GBDT before MLP; two-model prototype before shared-weight production build.

Each rung must beat the previous one to justify itself.

10.

Pre-commit tolerances and kill criteria.

Every phase states its acceptance numbers - and its rejection conditions - before results exist.

That is what separates an execution plan from a hopeful roadmap.

14.

Check your understanding.

Work these before peeking at the answers.

Q1.

Why must total U T steps be modified on the config object before from pretrained rather than on model dot config afterward? Q2.

Final-loop K V reuse at decode is near-lossless (G S M eight K 78.85 vs 78.92) while first-loop reuse collapses ( about 18.7), and any reuse during prefill costs more than 10 points.

Explain all three facts with one mechanism.

Q3.

In Phase 2, a batch of failing examples shows K L between consecutive loops falling below epsilon by loop 2, yet the answers are wrong.

What is happening, and which exit rule component protects against it? Q4.

State the correctness invariant of Phase 4's greedy self-speculative decoder, and explain why violating it indicates a bug rather than a quality trade-off.

Q5.

Your Phase 7 router keeps sending hard MATH five hundred prompts to T equals 2.

Which two config values would you adjust first? Q6.

Why does Phase 6 require calibrating each shared layer on activations aggregated across all loop indices, and what metric detects when this wasn't done properly? Q7.

Why is "loop to T equals 8 at inference for extra quality" not a viable plan for Ouro? Answers.

A1.

The remote modeling code consumes the loop configuration when the model object is constructed.

from pretrained with the modified config bakes the depth in; editing model dot config afterward changes a bookkeeping attribute the already-built forward path may never re-read.

Phase 0 centralizes the correct order in load Ouro so it can't be done wrong ad hoc.

A2.

The loop performs iterative refinement of the context representation.

During prefill, intermediate loops must read their own intermediate-quality K V to perform the refinement - collapsing caches corrupts the process itself (more than ten-point loss).

By decode time the prompt representation has converged, and the final loop's K V is that converged representation, so new tokens attending to it alone lose almost nothing.

First-loop K V is the unrefined draft - attending to it discards precisely the refinement the model's quality depends on.

A3.

The model has converged to a wrong attractor: the trajectory is stable but the fixed point encodes an incorrect answer.

Pure stability rules (K L, update norm) would exit early with confidence.

The conservative combined rule adds answer-level signals - parsed-answer stability plus confidence margin - and the phase's guidance is to validate every rule against the final task metric, never trajectory dynamics alone.

A4.

self-spec text must equal the full T equals four greedy text, token for token.

Every emitted token is either a draft token verified to equal the verifier's greedy choice at that position, or the verifier's own first token after a rejection - so by construction the output sequence is exactly what greedy T equals 4 decoding would have produced.

Any mismatch means the verification logic (positions checked, cache state, prefix acceptance) is implemented incorrectly.

A5.

Raise loss weight undercorrect relative to loss weight overcompute (making cheap-but-wrong routing more expensive during training), and raise minimum confidence so low-confidence predictions fall back to T equals 4.

If the misrouting persists, escalate the feature set from prompt-only to the one-loop probe, since hardness may only be visible after the model starts working.

A6.

A shared layer plays a different role at each loop index and sees a different activation distribution at each (LoopQ's "distribution shift across loop roles").

Calibrating on one role fits none of the others, and the resulting per-iteration error compounds recursively.

The detection metric is trajectory fidelity - cosine similarity between B F sixteen and quantized hidden states per (layer, loop) - with special attention to loop-transition fidelity, where recursive error spikes first; the acceptance gate demands mean fidelity greater than or equal to 0.98 with no catastrophic dips.

A7.

Ouro was trained at T equals 4, and quality peaks at the trained depth, degrading at T equals 5 - 8; scaling analyses independently show test-time loop gains saturate near the mean training recurrence.

Extra inference loops move the model off its training distribution rather than buying more refinement.

The exploitable direction is downward - spending fewer loops on easy inputs via exit rules and routing.

15.

Where to go next.

Within this folder.

The phase plans are execution-ready: start at phase zero, environment and baseline and respect the dependency graph in Section 3.

The R D T Guide supplies the surrounding field: Track B (converting a standard pretrained model to recurrent depth), the ecosystem or tooling status matrix, and the July 2026 open-problems watchlist - of which two directly gate this plan's ceiling: adaptive-exit-aware serving engines, and a packaged loop-aware quantization library.

Primary literature anchoring the phases (arXiv IDs as cited in the plans): Here are the entries in spoken form.

First: Topic: The model itself.

Paper: Ouro, also called LoopLM.

ID: 2510.25741.

Second: Topic: Recurrent-depth reference architecture.

Paper: Huginn.

ID: 2502.05171.

Third: Topic: Per-loop LoRA precedent (Phase 5).

Paper: Relaxed Recursive Transformers.

ID: 2410.20672.

Fourth: Topic: Constant-memory cache target (Phase 8).

Paper: MELT.

ID: 2605.07721.

Fifth: Topic: Loop-aware PTQ (Phase 6).

Paper: LoopQ.

ID: 2605.16343.

Sixth: Topic: Loop-aware G P T Q recipe (Phase 6).

Paper: Hyperloop Transformers.

ID: 2604.21254.

Seventh: Topic: Compression failure modes + fidelity metric (Phase 6).

Paper: "What Survives.Edge".

ID: 2606.26488.

Eighth: Topic: Stability + loop scaling ceiling (section 2.3).

Paper: Parcae.

ID: 2604.12946.

Ninth: Topic: Token-adaptive recursion context (Phase 7).

Paper: Mixture-of-Recursions.

ID: 2507.10524.

Tenth: Topic: Cross-loop parallel latency fix.

Paper: Parallel Loop Transformer.

ID: 2510.24824.

Community index: github.com or huskydoge or Awesome-Loop-Models - the field's de-facto tracker, with dated briefings.

Closing thought.

Nothing in this plan invents a new architecture.

Its value is the discipline: a fixed measurement foundation, an equivalence test guarding every intervention, controls that validate the harness, and pre-committed numbers deciding what ships.

The loop knob makes adaptive compute possible; the engineering method is what makes it trustworthy.

Lecture notes prepared July 7, 2026, from the materials in this folder: R D T Guide (snapshot date July 7, 2026) and Ouro execution phase plans phases 00 - 08.

Author-reported figures cited here (G S M eight K or MATH five hundred cache numbers, quantization collapse statistics) inherit the confidence labels of the source guide.
