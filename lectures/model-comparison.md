# Four Models, One GPU

### A technical lecture comparing recurrent-depth, dense, and mixture-of-experts language models on real hardware

*Approx. 20–25 minutes spoken. Written to be listened to — the numbers and the reasoning are spoken aloud rather than rendered in tables, so it reads cleanly through a text-to-speech engine. A companion to "Making Ouro Fast": that lecture was about squeezing a single recurrent-depth model; this one puts four very different models on the same GPU and asks a simpler question — which architecture actually earns its keep?*

---

## Cold open

Welcome back. Today I want to do something concrete. I want to take four language models that are built on genuinely different ideas, put them on the exact same graphics card, feed them the exact same questions, and measure what happens — not just whether they get the answers right, but what they cost you in memory, in latency, and in tokens per second. Because the marketing conversation about models is almost always about accuracy alone, and accuracy alone is a deeply misleading way to choose a model. The honest question is accuracy per gigabyte, accuracy per millisecond, accuracy for the hardware you can actually afford to run.

The four models are these. First, Ouro, a two-point-six-billion parameter recurrent-depth model from ByteDance — a model that loops its layers many times per token. Second, Huginn, a three-and-a-half-billion parameter recurrent-depth model from a university group — same core idea, different implementation. Third, Llama three-point-one, the eight-billion parameter instruct model from Meta — a conventional dense transformer, the workhorse baseline. And fourth, Qwen three, a thirty-billion parameter mixture-of-experts model that only activates about three billion parameters per token — the clever, sparse heavyweight.

So we have two recurrent-depth models, one dense model, and one mixture-of-experts model. Everything ran on a single RTX PRO six-thousand card, in sixteen-bit precision, one request at a time, on fifty questions each from three benchmarks: grade-school math, broad knowledge recall, and science reasoning. Let me take you through what we found, starting with the one architectural idea you have to understand before any of the numbers make sense.

---

## Part one — the looping trick, and why it changes the economics

Here is the strange thing that both Ouro and Huginn do, and that Llama and Qwen do not. A normal transformer runs each token through its stack of layers exactly once, top to bottom, and then predicts. A recurrent-depth model takes a small stack of layers and runs it in a loop, over and over, refining a hidden state each time, before it commits to an answer.

Ouro does this by taking its forty-eight physical layers and replaying the whole stack four times for every token. Four loops times forty-eight layers is one hundred and ninety-two effective layer-passes — that is the actual sequential compute each token goes through. Huginn does it a little differently: it has a small two-layer prelude that reads the input, then a four-layer core that it loops thirty-two times, then a two-layer coda that produces the output. Two, plus thirty-two times four, plus two, comes to one hundred and thirty-two effective layer-passes.

So both of these models are deep — well over a hundred layer-passes of compute per token. And here is the beautiful part: the weights are shared across every loop. Ouro is not storing one hundred and ninety-two layers of parameters; it is storing forty-eight and reusing them four times. Huginn is not storing a hundred and thirty-two; it stores eight and reuses the core. That is the entire trick, and it is why the memory story, which I'll come to, is so lopsided.

Now, why go to all this trouble? Because depth is where reasoning comes from. And we can watch it happen. When we sweep Ouro's loop count from one to two to four, its accuracy climbs monotonically. On the knowledge benchmark it goes from fifty percent at one loop, to sixty-eight percent at two loops, to seventy-four percent at four. On science reasoning, sixty-eight, then eighty-two, then eighty-six. On grade-school math, twenty-six percent, then sixty-two, then seventy. More loops, more thinking, more correct answers, every single time. Huginn shows the same shape on science reasoning — its accuracy rises from fourteen percent at eight loops, to twenty-six at sixteen, to thirty-four at thirty-two. The loop is not decoration. The loop is the reasoning, and you can turn it up like a dial.

Hold onto that: depth costs you speed, and buys you accuracy. Everything downstream is a negotiation over that trade.

---

## Part two — the accuracy scoreboard, told honestly

Let me give you the headline accuracy numbers, and then immediately complicate them, because the complication is the whole point.

On grade-school math, Qwen, the thirty-billion mixture-of-experts, wins at ninety percent. Llama, the eight-billion dense model, gets eighty. Ouro, at two-point-six billion, gets seventy-two. Huginn is very weak here, near two percent — I'll explain that in a moment. So on raw math, the bigger conventional models win. No spin: Ouro trails them on arithmetic reasoning.

But now look at knowledge recall. Qwen gets seventy-six percent. Ouro gets seventy-four. Llama, the eight-billion model, gets only sixty-six. Read that again — the two-point-six-billion recurrent model beats the eight-billion dense model on broad knowledge, and lands within two points of the thirty-billion mixture-of-experts. And on science reasoning it is even starker: Ouro scores eighty-six percent, beating Llama's seventy-four handily, and trailing Qwen's ninety-two by only six points, at roughly a twelfth of the active-parameter budget of the comparison.

So the fair summary of Ouro is not "it wins." It is "it punches wildly above its weight." It is competitive-to-winning against a model three times its size, and within a few points of one more than ten times its size, on everything except pure math. That is the accuracy story: strong-for-its-size, not dominant.

Huginn is the honest disappointment on task accuracy — around thirty percent on knowledge, thirty-four on science, and effectively zero on grade-school math in our zero-shot setup. And that is real, not a bug; it matches the paper's own reporting. Huginn is a research model whose gains on knowledge tasks are modest and whose math ability really needs few-shot prompting to show up. So why keep it in the comparison at all? Because Huginn is not here to win on accuracy. Huginn is here for the second half of this lecture — it is the vehicle for showing that these looping models are not just efficient, they are improvable. Keep it in your pocket.

---

## Part three — the memory headline, which is the whole argument

Now the hardware, and this is where the recurrent-depth models stop being an underdog story and start being an obvious win.

Let me give you four numbers, all peak memory during generation. Ouro, at its full four-loop depth, uses about six gigabytes. Huginn uses seven to ten, depending on how many loops. Llama, the eight-billion dense model, uses just over fifteen gigabytes. And Qwen, the thirty-billion mixture-of-experts, uses fifty-seven gigabytes.

Sit with that. Ouro is roughly two-and-a-half times lighter than the eight-billion Llama, and about nine-and-a-half times lighter than the thirty-billion Qwen — while matching or beating both on knowledge and reasoning. That is the shared-weight trick paying off in the most direct way imaginable. A hundred and ninety-two layer-passes of compute, from six gigabytes of footprint, because it is forty-eight layers wearing four different hats. You could run Ouro comfortably on a consumer card that would not even load Qwen.

The prefill latency — the time to digest the prompt before the first token appears — tells a related story. Ouro, Huginn, and Llama all prefill in a comfortable twenty to seventy milliseconds. Qwen pays five hundred and twelve milliseconds. Half a second, just to read the prompt, before it says a single word. That is the mixture-of-experts tax: the model is enormous and its routing machinery is expensive to spin up, and it shows up as latency even on the benchmarks where Qwen wins on accuracy.

Now, the recurrent-depth models do pay for their depth in raw decode speed, and I won't hide it. Llama, being a shallow single-pass model, is the fastest decoder at about sixty-eight tokens per second. Ouro at four loops does about twenty-one. Huginn at thirty-two loops does about ten. The loop tax is real. But notice — Qwen, despite ten times the parameters, only decodes at about seventeen tokens per second, barely faster than Ouro, because its half-second prefill and its size drag it down. So the picture is not "conventional models are fast, recurrent models are slow." The picture is: Llama is fast and heavy, Qwen is slow and enormous, and Ouro is a little slower than Llama at decode but a fraction of the memory of either. For most real deployments, where memory is the constraint that actually stops you, that trade is very attractive.

So the memory argument, in a breath: recurrent depth gives you deep reasoning from a tiny, shared parameter footprint. That is the efficiency-per-parameter case, and it is strong.

---

## Part four — the improvable story, part one: accelerating without losing quality

Here is the pivot. If a recurrent-depth model's only weakness is decode speed, and decode speed comes from the loop, then the interesting question becomes: can you make the loop cheaper without giving up the accuracy the loop buys you? And the answer, it turns out, is yes — several different ways. This is the "improvable" story, and it is the reason to be excited about this architecture rather than just impressed by it.

Let me walk you through the accelerators, from the boring-but-free to the genuinely surprising.

The first is exact kernel optimization. This is unglamorous plumbing: you take all the tiny operations inside each loop — the separate query, key, and value projections, the separate parts of the feed-forward block, the normalization steps — and you fuse them into fewer, larger GPU operations. Same arithmetic, exactly the same output, just packed into fewer kernel launches so the graphics card spends less time idling between tiny jobs. On Ouro this gives about a one-point-three-six times speedup, and it is bit-exact — the output is provably identical. A free lunch, if an undramatic one.

The second is prompt-lookup speculation. This one is clever and almost embarrassingly cheap. As the model generates, a lot of what it is about to say has already appeared — a number from the question, a name, a repeated phrase. So instead of generating those tokens one expensive loop at a time, you draft them by copying chunks of text the model has already seen, and then you verify the whole draft in a single forward pass. When the draft is right, you got several tokens for the price of one. When it is wrong, you fall back. It is lossless — you only ever accept tokens the full model agrees with — and on the math benchmark it is the fastest Ouro path, about one-point-six-four times faster.

The third is depth-speculative decoding, and this is the one that generalizes to the loop directly. The idea: drafting a token at full depth is expensive, so draft it cheaply at shallow depth — one loop instead of four — and then verify the drafted block at full depth in one pass, committing whatever prefix the full-depth model agrees with. You get full-depth quality, because the full-depth model is the one that signs off, but you pay mostly shallow-depth cost. On Ouro this gives about one-point-four-one times; on Huginn, drafting at eight loops and verifying at thirty-two, also about one-point-four-one. Quality-preserved, meaningfully faster.

So already, before we do anything exotic, recurrent depth is accelerable in the range of one-point-three-six to one-point-six-four times, with the answer either provably identical or verified by the full model. That alone answers the "is it improvable" question. But the last accelerator is a different order of thing entirely, and it is worth slowing down for.

---

## Part five — the improvable story, part two: sampling the loop in parallel

Everything so far has still been fundamentally sequential: one token, then the next, then the next. The last idea breaks that assumption, and it comes from a lovely observation in a recent paper — that a recurrent-depth model, refining a hidden state through many steps, is mathematically close cousin to a diffusion model, which refines a noisy signal through many steps. And diffusion models can be sampled in parallel.

So here is the sampler, called the efficient parallel sampler, or the diffusion-forcing sampler. Instead of finishing all the loops for one token before starting the next, you maintain a whole moving front of tokens at once — a wavefront. Every forward pass, you advance the loops on all of the active tokens together, and you let each one keep refining. As soon as a token at the left edge of the wavefront settles — as soon as its hidden state stops changing much between steps — you freeze it, commit it to the output, and add a fresh token at the right edge. The compute is still recurrent, still deep, but instead of a long chain of tiny one-token steps, the graphics card sees a wide batch of positions and can work on them all at once. Small-batch decoding, which normally wastes a GPU, suddenly becomes a fat, efficient matrix problem.

And when we ran the paper authors' own implementation of this sampler on Huginn, here is what happened. The ordinary sequential decoder, at thirty-two loops, ran at about ten tokens per second. The parallel diffusion sampler ran at about forty-four tokens per second — a four-point-two-four times speedup — and it produced coherent output, matching the sequential decoder token-for-token on several of the test problems and staying close on the rest, exactly as an approximate-but-verified parallel sampler should. Four times faster, same quality. And that number, four-point-two-four, lands right on top of the four-point-three-six the paper reports for the same benchmark. It reproduces.

Now let me be scrupulously honest about the road here, because it is a good lesson. Our own first attempt to build this sampler from scratch failed — the parallel tokens collapsed into repetitive garbage. And chasing that down taught us exactly why the method is subtle: it needs a very specific cache strategy for handling tokens that are at different stages of refinement, and it needs each new token to start from a fresh random state rather than a copy of its neighbor, or the whole wavefront collapses into a symmetric heap. Once we ran the authors' actual code, with their actual cache, it worked immediately and reproduced their result. The lesson is not "we failed." The lesson is that the four-times speedup is real and reproducible, and that the details of a research method are load-bearing — you cannot eyeball them.

---

## Part six — the verdict

So let me put the whole comparison in one frame.

On accuracy, the bigger conventional models win on pure math, but the two-point-six-billion recurrent model beats the eight-billion dense model on knowledge and science, and comes within a few points of a thirty-billion mixture-of-experts — at a fraction of the size. On memory, it is not close: the recurrent models run in six to ten gigabytes where the dense model needs fifteen and the mixture-of-experts needs fifty-seven. And on the question of whether the architecture's one real weakness, decode speed, can be fixed — the answer is a clear yes, from one-point-three-six times with a free bit-exact kernel, to one-point-four-one with depth-speculation at preserved quality, all the way to four-point-two-four times with a parallel diffusion sampler at coherent quality.

The honest positioning of recurrent-depth models is therefore not "they beat everything." It is two things at once. First, efficiency per parameter: they reach the reasoning depth of a much larger model from a tiny shared footprint. And second, accelerability: their single weakness, the sequential loop, is exactly the thing that a growing family of samplers knows how to speed up — because a looping model, it turns out, is a diffusion model wearing a transformer's clothes.

If you take one thing from this, take that pairing. Not raw superiority — efficiency, plus a real and reproducible path to being made several times faster without giving up the answers. That is a very good place for an architecture to be. Thanks for listening, and I'll see you in the next one.
