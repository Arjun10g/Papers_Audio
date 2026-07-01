# Making Ouro Fast
### A technical lecture on optimizing a recurrent-depth model for GPU inference
*Approx. 20–25 minutes spoken. Written to be listened to — the numbers and the reasoning are spoken aloud rather than rendered in tables, so it reads cleanly through a text-to-speech engine. A hardware companion to "Thinking in the Dark": that lecture is about why recurrent-depth models think the way they do; this one is about making them run fast without changing a single answer.*

---

## Cold open

Welcome back. Today I want to walk you through a piece of systems work: the effort to make a recurrent-depth language model run faster on a GPU, without changing a single one of its answers. That last part is the whole discipline. It is easy to make a model faster if you are allowed to make it a little bit wrong. The hard, honest game is to make it faster while proving, at every step, that the output is exactly what it was before.

The model at the center of this is Ouro, ByteDance's family of recurrent-depth models. And to understand why the optimizations look the way they do, you first have to understand the one strange thing Ouro does differently from an ordinary transformer. So let's start there, and then I'll take you through what we built, in the order we built it, and where the work goes next.

---

## Part one — why a looping model is expensive

Here is the core idea. A normal transformer runs each token through its stack of layers exactly once. Ouro does not. Ouro takes its stack of layers and runs it in a loop, several times, for every single token, before it emits anything. The config has a knob for this called total-u-t-steps, the number of universal-transformer steps, and you can set it to one, two, four, and so on. So when we talk about a ninety-six-layer effective forward pass at four steps, what is physically there is twenty-four layers, replayed four times over.

Now, why does that matter for hardware? Two consequences, and they drive everything.

The first consequence is the key-value cache. In any transformer, as you generate, you cache the keys and values from previous positions so you don't recompute them. But because Ouro replays its layers, the effective number of cache layers is the number of loop steps multiplied by the number of physical layers. For the one-point-four-billion parameter Ouro at four steps, that is ninety-six cache layers. Concretely, that works out to about seven hundred and eighty-six kilobytes of cache for every single token, which climbs to roughly three gigabytes once you are a few thousand tokens into a context. The loop multiplies your memory footprint.

The second consequence is latency, and this is the one we measured directly, so hold onto it. Decode time — the time to produce each new token — scales almost linearly with the loop depth. At one loop step, producing a token took about thirteen milliseconds. At two steps, about twenty-five. At four steps, about forty-nine. The loop is not a detail. The loop is the cost.

So the whole project is, in one sentence: pay that loop tax more cheaply, and never change the answer while doing it.

Hold onto that framing. Everything from here follows from it.

---

## Part two — the method, which matters more than any single trick

Before I show you a single optimization, I want to talk about how we work, because the discipline explains the shape of everything else.

The first principle is exact-first, approximate-later. The default path is exact. Every optimization has to first prove it produces identical output — the same generated tokens, and ideally the prefill logits are bit-for-bit identical, a maximum absolute difference of exactly zero — before it is allowed to count as done. Only after the exact paths are built and measured do we allow ourselves to reach for approximate tricks, like quantization or cache reuse or parallel sampling. And when we do, those live behind an explicit switch, with a way to fall straight back to exact. A speedup that is bit-exact is a free speedup. A speedup that is only approximately correct is a quality-gated speedup, and it is never, ever allowed to become the default silently. You will hear that distinction over and over.

The second principle is that every optimization is reversible. We apply changes as patches — little functions that wrap or swap parts of the model — and every patch has a matching restore function, or it is simply a command-line flag you can choose not to pass. Nothing rewrites the model permanently. There is always a documented way back.

The third principle is about the hardware itself. GPU runs go out to a rented instance on a marketplace — mostly an RTX PRO six-thousand Blackwell card, some earlier runs on an H100. And the rule there is strict: rent the machine, run one bounded job, and then tear the instance and its temporary key down immediately, in a cleanup block that runs whether the job succeeded or failed. We track the cost to the cent. Local development uses tiny models on the CPU so that nothing paid ever runs by accident.

And the fourth principle is simply: write everything down. Every phase records its goal, the files it changed, the exact commands, the verification result, the numbers, the rollback path, and the follow-ups. That written ledger is the reason we can trust any of the numbers I'm about to give you.

So the rule, in a breath: prove it's exact, keep it reversible, measure it on a machine you immediately give back, and log all of it. The kernels are almost the easy part.

---

## Part three — build the ruler before you cut

You cannot optimize what you cannot measure, so the very first work was not a kernel at all. It was a measurement harness.

There are two halves to the codebase. One half is the performance library, which knows how to time things and count things: it measures time-to-first-token, time-per-output-token, end-to-end latency, tokens per second, peak GPU memory, and it has an estimator for how many bytes the key-value cache will consume. The other half is the optimization library, which holds the actual patches — and at the start it was mostly empty, waiting to be filled one phase at a time.

Let me give you the vocabulary you'll need for the rest of this lecture, because I'll lean on three numbers. Time-to-first-token is dominated by prefill, the processing of the prompt. Time-per-output-token is dominated by decode, the generation loop — and that is where the loop depth hurts. And end-to-end is just the whole thing, start to finish.

The point of this phase was to build a trustworthy ruler and prove it read correctly. As a sanity check, the cache estimator predicted seven hundred eighty-six thousand four hundred and thirty-two bytes per token for the one-point-four-billion model at four steps — the same number I quoted you earlier, now coming out of the tool rather than off a napkin.

---

## Part four — the measurement that set the whole agenda

With the ruler built, we ran the model across a grid: a few prompt lengths, a couple of output lengths, and loop depths of one, two, and four. Fifty-four measurements on an H100, and then, as always, we deleted the machine.

And here is the result that shaped everything after it. Time-per-output-token scaled almost perfectly with loop depth: about thirteen milliseconds at one step, twenty-five at two, forty-nine at four. Meanwhile, the prompt length barely moved the per-token decode time at all — longer prompts mostly cost more at prefill and in memory, not in the steady-state generation loop.

Think about what that tells you. The profitable place to cut is the per-decode-step work, the thing that gets replayed once for every loop step. That is the cache growth, the linear projections, the normalization layers, and the sheer overhead of launching kernels. That ranking — attack the decode step, leave prefill alone — is exactly the order the rest of the work follows.

---

## Part five — the exact optimizations, one at a time

The plan lists the exact optimizations in a deliberate order: first a flat cache, then packed projections, then fused normalization, then CUDA graphs, and finally, still ahead of us, an exact paged cache. Let me tell you the story of each, because each one taught us something.

### The flat cache

Start with the cache. The standard Hugging Face cache grows its key and value tensors by concatenation — every new token, it glues a slice onto the end, which quietly reallocates and copies the whole thing. Because Ouro loops, that happens a staggering number of times. In one short baseline workload, we counted fourteen hundred and forty of those concatenation calls.

The fix is to stop growing and start writing into place. We preallocate one big arena, sized to the full prompt-plus-output length up front, and each step simply writes into the correct slice of it. No concatenation, ever.

And here is the honest part of the story, the part I want you to remember. The correctness was perfect on the first try: the generated tokens matched, the logits differed by exactly zero, and those fourteen hundred and forty concatenation calls dropped to zero. But the first version was slower. Slower, despite doing less allocation, because the Python-side bookkeeping and the strided views into that big preallocated block cost more than they saved. It was correct immediately and fast not at all.

So there was a grind — several sub-phases of it. We built a dedicated micro-benchmark just for the cache-update hot path, so we could iterate cheaply in isolation instead of spinning up the whole model. We rejected one tempting shortcut that would have compromised exactness. And we ground the write path down with more direct views and lower-level indexing, tracked in a whole series of before-and-after measurements. That is the real texture of optimization: exactness is a gate you pass once, but performance is a hill you climb over several attempts.

### Packed projections

Next, the projections. In each decode step, the model does separate matrix multiplies for the query, the key, and the value — and separately again for the two halves of its gated feed-forward network, the gate and the up projection. On a single-token decode, each of those separate matmuls is its own kernel launch, and the overhead of launching them dominates the tiny amount of actual math.

So we pack them. Concatenate the weight matrices, do one larger matmul, then slice the result back apart. We did this first for the query-key-value projection, then for the feed-forward gate and up projection, and then combined both in one full-model test.

The result was a clean, exact win: roughly seven percent faster decode, with the generated tokens and prefill logits still bit-identical. But there was a catch worth naming, because it is a consequence of our own rules. Peak memory went up by about one-point-six gigabytes. Why? Because the reversible patch keeps the original separate weight matrices registered alongside the new packed ones, so that you can always roll back and so the saved-model format stays compatible. Reversibility is not free — you are, in effect, keeping the old parts in the drawer so you can always put them back. A follow-up that is not reversible could reclaim that memory once we trust the exactness enough to stop hedging.

### Fused normalization

Now the normalization layers. Each decoder layer normalizes twice, and each normalization is paired with a residual add. On a single-token decode, that is a swarm of tiny elementwise operations, each one too small to keep the GPU busy. The idea is to fuse them into a single custom kernel.

We built this in two stages. First a scaffold: a plain reference implementation in the framework's own math, matching Ouro's normalization exactly, plus optional hand-written Triton kernels for the standalone norm and for the fused residual-plus-norm. On the kernel in isolation, the fused Triton version was about two-point-three times faster than the naive path.

Then we promoted it to a full-model test with strict parity checks — and this is the most instructive result in the whole project, so let me slow down. There were two backends. The framework-math backend was bit-exact: identical tokens, logits differing by exactly zero. But it gave no real speedup at the full-model level. The Triton backend was the opposite: it produced the same generated tokens and it was about twenty-five percent faster end-to-end — but its prefill logits were not bit-exact. They differed by about one-eighth in absolute terms.

Under our rules, that difference is disqualifying for a default path. So the fast kernel does not get promoted. It survives only as an opt-in, quality-gated branch, something you can turn on deliberately if you have decided the small numerical drift is acceptable for your use. The exact framework-math version, meanwhile, gets carried forward as a parity-proven, reversible hook — not as a performance win, but as a correct building block.

This is the clearest illustration of the entire philosophy. A twenty-five percent speedup that is not bit-exact does not get to be the default. Speed never quietly overrules correctness.

### CUDA graphs

The last of the finished work, and the biggest decode win so far, is CUDA graphs. Even with fused kernels, launching hundreds of GPU operations per decode step from Python carries overhead — and remember, Ouro pays that overhead once per loop step. A CUDA graph records a fixed sequence of GPU operations one time, and then replays the whole thing with a single launch, erasing the per-launch cost. The price of admission is rigidity: a graph demands static shapes and static memory addresses, which is exactly what variable-length text generation does not naturally give you.

So this phase climbed a ladder of increasing realism, five rungs.

First, a synthetic gate: prove the mechanism on a fake Ouro-shaped decode loop. The replay was exact, and about one-point-three times faster. That established that capture and replay work at all, and that they have to be bucketed by shape.

Second, a full-model, single-token feasibility test on the real one-point-four-billion model. Exact replay again, and now about two-and-a-half times faster on the decode. This rung also surfaced a real discovery: the ordinary all-ones attention mask breaks graph capture inside the framework's mask-creation code — it hits an operation that CUDA graph capture simply does not support. So we switched to a no-padding decode bucket, and we keep the padded-mask path flagged as known-unsafe.

Third, an advancing multi-token chain. Rather than one token, we generate a short chain by capturing one graph per fixed cache position, and feeding each step's chosen token into the next. Exact chain, about two-and-a-half times faster again.

Fourth, we refactored that prototype out of the benchmark and into a real, reusable generation helper, with proper prompt-length and output-length bucket metadata. On the bucket we tested, this reached about three-point-seven times faster on decode. This rung also caught a measurement bug in ourselves — the graph timing had accidentally been including a reference check, which we corrected by snapshotting the replay time before the check runs. Worth saying out loud: measuring honestly includes catching your own measurement mistakes.

And the fifth rung, which is in progress right now, extends that single bucket into a sweep across many prompt and output lengths, so we can study how the one-time cost of capturing a graph gets amortized as the output grows.

So the summary on CUDA graphs: it is the largest exact decode win we have, somewhere between two-and-a-half and three-point-seven times, but it is bucketed and it is fragile. Static shapes only, no-padding masks only, and right now, one captured graph per decode position. A single reusable graph with a dynamically driven cache-write offset is still unproven — and that is precisely the open research edge.

---

## Part six — where we stand

Let me give you the scoreboard in plain words. Every phase I'm about to name cleared the exactness bar unless I say otherwise.

The benchmark harness and the baseline matrix are done — and they are what told us the loop depth is the enemy. The flat preallocated cache is done and exact; it took the concatenation calls from fourteen hundred and forty down to zero, and its performance was tuned over several follow-on phases. The packed query-key-value and feed-forward projections are done and exact, buying about seven percent on decode at the cost of some extra memory from staying reversible. The normalization work is done, with the framework-math version exact and the faster Triton version deliberately held back behind a quality gate. And the CUDA graph decode replay is done and exact on no-padding buckets, worth somewhere from two-and-a-half to three-point-seven times on decode. The one piece still moving is the bucket sweep on top of the graph work.

A few standing decisions anchor all of this. Exact behavior comes before any kernel work or any approximate shortcut. The cache math is always the loop steps times the layer count. Rented machines are always bounded and always torn down immediately. And we pin the framework versions so that a run today is comparable to a run last week.

---

## Part seven — what comes next

So where does the work go from here? Let me lay out the road.

The immediate work is finishing the CUDA graph story. That means completing the bucket sweep with repeated runs, so we can quantify how the capture cost amortizes over longer outputs. It means proper replay-only profiling — traces that cleanly separate the pure graph-launch time from the token-feedback and the device-to-device copies, because our current profiler numbers still fold in some setup overhead. It means the hard one: replacing one-graph-per-position with a single reusable graph whose cache-write offset is driven by a tensor, which is still unproven. And it means making the padded-mask path safe for capture, so that batched, padded decoding can also use graphs, not just the clean no-padding case.

After that, there is one more exact optimization on the list: an exact paged cache, extending the flat cache toward a block-structured layout.

And then, and only then, we open the door we have kept shut on purpose: the approximate, quality-gated frontier. That is where we resolve whether the faster Triton normalization can be made acceptable with broader testing and a tolerance gate. It is where residual fusion gets a proper decoder-layer integration test. It is where quantization experiments live. And it is where the biggest potential payoff sits: efficient parallel sampling, a way of producing several tokens at once instead of strictly one at a time. That one is deliberately gated until the exact baseline, the flat cache, at least three exact kernel wins, and a clean exact rollback report are all in place. The default fallback for it is always plain, exact, one-token-at-a-time generation.

There are risks we carry openly. Every optimized path has to keep re-proving its exactness against the baseline. The faster normalization and the residual fusion are not yet exact or not yet fully integrated. The CUDA graphs are validated only on the clean no-padding case. And serving through the big inference engines, with their trust-remote-code and fixed-loop-count requirements, needs its own separate baseline before we can claim anything there.

---

## Recap, in five breaths

Ouro loops its layers several times per token, so both the decode cost and the cache size scale with the loop depth — and that single fact is the entire problem.

We built the measurement harness before anything else, and it told us to attack the decode step and leave prefill alone.

Then we did decode surgery in a strict order — flat cache, packed projections, fused normalization, CUDA graphs — and every one of them had to prove it was bit-exact and reversible before it counted.

The current frontier is CUDA graph generation, worth roughly two-and-a-half to three-and-a-half times on decode, but bucketed and fragile, with a sweep in progress.

And the tempting approximate speedups — the faster normalization, quantization, parallel sampling — are real, and they are deliberately locked behind quality gates until the exact roadmap is finished. Because a fast wrong answer is not an answer.

That's the state of the work. Thanks for listening.
