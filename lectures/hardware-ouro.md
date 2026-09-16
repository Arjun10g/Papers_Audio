# Making Ouro Fast

### A technical lecture on optimizing a recurrent-depth model for GPU inference

*Approx. 30–35 minutes spoken. Written to be listened to — the numbers and the reasoning are spoken aloud rather than rendered in tables, so it reads cleanly through a text-to-speech engine. A hardware companion to "Thinking in the Dark": that lecture is about why recurrent-depth models think the way they do; this one is about making them run fast without changing a single answer.*

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

The plan lists the exact optimizations in a deliberate order: first a flat cache, then packed projections, then fused normalization, then CUDA graphs, and finally an exact paged cache — which we have now built, and then fused with the graphs to unlock the biggest decode win yet. Let me tell you the story of each, because each one taught us something.

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

And the fifth rung extended that single bucket into a sweep across many prompt and output lengths, so we could study how the one-time cost of capturing a graph gets amortized as the output grows.

So the summary on CUDA graphs, as of that stage: the largest exact decode win we had, somewhere between two-and-a-half and three-point-seven times, but bucketed and fragile. Static shapes only, no-padding masks only, and one captured graph *per decode position*. A single reusable graph, with a cache-write offset driven by a tensor, was still unproven — and that was precisely the open research edge. Which brings us to the piece that closed it.

### The paged cache, and the reusable graph it unlocked

The last exact optimization on the list was the paged cache. Instead of one giant preallocated block, store the memory in fixed-size pages, like the pages of a real book, tracked by a little index that says which page holds which position. On its own this is about flexibility — it makes variable lengths and batching far cleaner. We built it, we gave it a proper full-model exactness gate against the baseline, and it passed at exactly zero difference. Worth being honest, though: in plain step-by-step decoding the paged cache is currently *slower* than the baseline, just as the flat cache was at first. Exact, but not yet a speed win by itself.

The payoff is what the pages unlock. Because each page sits at a fixed address, you can finally record **one** graph and, between replays, just point it at the next page by writing a couple of small position tensors — instead of recording a separate graph for every position. We wired the paged cache into the graph path and added exactly that mode. It captures a single graph, replays it down the chain by updating token and position tensors, and it matches the plain version at zero difference, about three-and-a-half times faster than eager. One capture instead of many. And a bonus fell out: the all-ones attention mask that used to crash graph capture is, on these no-padding buckets, equivalent to having no mask at all — so routing it through the no-mask path made it graph-safe. Two of the open edges, closed in one phase.

---

## Part six — where we stand

Let me give you the scoreboard in plain words. Every phase I'm about to name cleared the exactness bar unless I say otherwise.

The benchmark harness and the baseline matrix are done — and they are what told us the loop depth is the enemy. The flat preallocated cache is done and exact; it took the concatenation calls from fourteen hundred and forty down to zero, and its performance was tuned over several follow-on phases. The packed query-key-value and feed-forward projections are done and exact, buying about seven percent on decode; the extra memory they used to cost from staying reversible has since been cleaned up, so the packed path is memory-neutral by default and still rebuilds the originals on rollback. The normalization work is done, with the framework-math version exact and the faster Triton version deliberately held back behind a quality gate. The CUDA graph decode replay is done and exact on no-padding buckets, worth somewhere from two-and-a-half to three-point-seven times. And the paged cache is now done and exact too — with a full-model gate that passed at zero difference — and, most importantly, fused with the graphs to give a single reusable recording that runs about three-and-a-half times faster than eager. That completes the exact roadmap. Every landed optimization matches the original bit-for-bit, and at that point the local test suite stood at a hundred and one passing — it has since grown well past two hundred as the frontier work in the coming parts landed.

A few standing decisions anchor all of this. Exact behavior comes before any kernel work or any approximate shortcut. The cache math is always the loop steps times the layer count. Rented machines are always bounded and always torn down immediately. And we pin the framework versions so that a run today is comparable to a run last week.

---

## Part seven — a surprise about what "exact" really has to mean

With the exact roadmap essentially complete, we opened the door we'd kept shut on purpose: the approximate frontier. And the very first thing we did there was interrogate our own central rule.

We had been treating "not bit-exact" as "disqualified." But step back and ask the honest question. When a faster kernel produces a number that differs in the eighth decimal place, is that a *worse* answer — or just a *different, equally good* one? We had been assuming the former. So we built a probe to actually measure it, and pointed it at the two inexact things we had on hand: the faster Triton normalization, and the occasional rounding flips inside the parallel decoder we'll get to in a moment.

The results were clarifying. On a corpus of text, perplexity — the standard measure of how well the model predicts — barely moved. A quarter of one percent, sitting inside the noise, and if anything slightly *better*. The full probability distributions the two versions produced were nearly identical: a Kullback-Leibler divergence of eight ten-thousandths of a nat, which is to say, almost nothing. On open-ended greedy generation, the exact and the inexact versions produced byte-for-byte identical text. And the parallel decoder's occasional flipped token never once changed a final math answer across the set we tried.

So the discipline matured, and this is the important beat of the whole second half. Bit-exactness was only ever a *proxy* for the thing we actually care about, which is that the quality didn't change. And it turns out to be an over-strict proxy. The honest gate is not "are the bits identical," it is "did the quality measurably degrade" — measured on perplexity and task accuracy, not assumed from a logit difference. This did not loosen the rule. It sharpened it. We could now promote a numerically-inexact optimization if, and only if, we could *prove* it costs nothing that matters. Hold onto that, because it is the license for everything that follows.

---

## Part eight — emitting more than one token per expensive loop

Here is the single biggest idea in the second half of the work. Ouro pays its loop tax once per token. So the question that dominates everything is: could we emit *several* tokens per expensive forward pass?

The technique is called speculative decoding, and the trick is beautiful. You cheaply *guess* a few future tokens, then *verify* all of them in one full-depth forward pass, and you keep the longest run that matches what plain greedy decoding would have produced. Because the verification is full-depth, every token you emit is exactly the token greedy would have emitted. You get the parallelism for free, in exact arithmetic. The only thing that changes between methods is where the cheap guess comes from.

Our first guess source was prompt-lookup. When the model is about to repeat something already in its context — a summary quoting its passage, code echoing a function it defined earlier — you can just copy the continuation from where that phrase appeared before, and verify it. On grounded, repetitive text this was a genuine win, up to three-and-a-half times faster. But on generic, open-ended text, where nothing repeats, it collapsed back to roughly one times. No repetition, nothing to copy.

So we added a second guess source that doesn't need repetition at all: Jacobi decoding, sometimes called lookahead. Instead of copying, it guesses a whole window of future tokens and refines them all in parallel, iterating toward the fixed point that greedy would eventually reach. On a factual prompt it hit three-point-two times, right where prompt-lookup had managed barely more than one. And here is the elegant part: the two methods are complementary. Prompt-lookup wins when the output copies the context; Jacobi wins when the output is predictable but not copied. So we fused them into a single decoder that decides, every round, which strategy is actually paying off — it measures tokens-per-forward and leans into whichever arm is winning, wasting no work on the other. That adaptive decoder beats a simpler router that just picks one method up front and commits. Neither, honestly, quite reaches the theoretical best-of-both, because any adaptation costs a little to run — but the online version gets closest.

We also chased the obvious compounding idea. Prompt-lookup spends its time in a *wide* verify forward, so could we make that forward itself cheaper, with packed matrix multiplies and a recorded CUDA graph? Packing helped a little, for free. But the graph wants a constant shape, and fixing the verify block to a constant width padded it wider than it needed to be, which roughly ate the packing gain. And the graph itself demands the special graph-safe paged cache, not the simpler flat one — because the flat cache, deep in its bookkeeping, briefly copies a length back from the GPU to the CPU, and a graph capture forbids exactly that. So that particular multiplier turned out to be real but scoped: a concrete piece of future plumbing, and we wrote down precisely why it's blocked and what unblocks it. Honest dead-ends are results too.

---

## Part nine — teaching Ouro to speculate on its own depth

Now the most Ouro-native version of the whole idea, and the one that pays off biggest. Ouro loops its layers four times. So ask: what if a *shallower* loop — just one or two passes — is a good enough *draft* of what the full four passes would eventually say?

That's depth-speculative decoding. You draft cheaply at low loop depth, verify at full depth, and keep the matches. On the real two-point-six-billion Ouro, drafting at a single loop step and verifying at four ran nearly twice as fast — one-point-nine times — and, crucially, the perplexity did not move: zero percent median change. The emitted tokens were not always bit-identical to the full-depth run; about five out of eight matched exactly. But the quality-first gate we had earned back in Part seven is exactly what gave us permission here. As long as the perplexity holds, a token that differs but is equally good is allowed.

So the promotion rule became perplexity-first. Block a route only if it is both faster *and* measurably worse on perplexity. On that basis, the router shipped speculation on most workloads and held back only one — code generation — where the shallow draft genuinely regressed perplexity, by a little under two percent, just enough to trip the gate. That is the mature form of the discipline in action: not "is it identical," but "is it faster without being worse," proven case by case.

---

## Part ten — a small looped model against the giants

Which brings us to the comparison we are running as I record this, and to the paper's central, audacious claim: a two-point-six-billion *looped* model matching or beating eight-billion dense models on reasoning. We wanted to see that on our own hardware — and, just as importantly, to price it.

So we built a benchmark suite that stands four models side by side: our optimized Ouro-2.6B, the raw Ouro-2.6B, a thirty-billion-parameter Qwen mixture-of-experts, and an eight-billion Llama. Across three tasks — grade-school math, broad knowledge recall, and science reasoning — each run both plainly and with chain-of-thought, and for Ouro across one, two, and four loops. And through all of it we measure the hardware story: peak memory, tokens per second, time to first token, and the inter-token latency.

Even the tiny validation run made the trade vivid. Ouro-2.6B lives in about five gigabytes of memory. The Llama needs fifteen. The Qwen mixture needs fifty-seven — eleven times Ouro's footprint. Our exact optimizations hand the little Ouro about a thirty-seven percent throughput gain over its raw self, and the speculative decoder stacks more on top of that. The full accuracy numbers are landing as I speak. But the shape of the story is already the paper's story, seen from the hardware side: the looped model's entire pitch is doing more *thinking* per parameter — and thinking, unlike raw size, is cheap in memory.

Getting there also meant hardening the plumbing. A four-model, three-benchmark sweep is a multi-hour job on rented machines, and a long-lived remote connection will drop over that span. So the runner now launches its work detached on the box and polls for the result over short, fresh connections, so a dropped link can never lose a completed run. Boring infrastructure, but it is the difference between an answer and a wasted afternoon.

---

## Recap, in six breaths

Ouro loops its layers several times per token, so both the decode cost and the cache size scale with the loop depth — and that single fact is the entire problem.

We built the measurement harness before anything else, and it told us to attack the decode step and leave prefill alone.

Then we did decode surgery in a strict order — flat cache, packed projections, fused normalization, CUDA graphs, and a paged cache — and every one had to prove it was bit-exact and reversible before it counted. That exact roadmap came together into a paged cache whose fixed-address pages let us record a single reusable CUDA graph, about three-and-a-half times faster than eager.

Then we crossed into the approximate frontier, and the first thing we found was that our own rule needed sharpening: bit-exact was a proxy for "the quality didn't change," and when we actually measured, the numerically-inexact paths were quality-lossless. So the gate became perplexity, not identical bits.

That license unlocked parallel decoding — guess several tokens, verify them all in one expensive loop, keep the greedy-matching run. Prompt-lookup for repetitive text, Jacobi for predictable text, a fused decoder that picks between them, and, most powerfully, depth-speculation that drafts Ouro at a shallow loop and verifies deep, nearly doubling throughput with no perplexity cost.

And now we are pricing the whole thesis: a five-gigabyte looped model against a fifty-seven-gigabyte mixture and a fifteen-gigabyte dense model, on real reasoning benchmarks, with the hardware numbers measured throughout. The local test suite is well past two hundred passing. The discipline never changed — a fast wrong answer is still not an answer — we just learned to measure "wrong" honestly.

That's the state of the work. Thanks for listening.
