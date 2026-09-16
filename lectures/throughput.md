# Same Model, Twice as Fast

### How to optimize an existing Transformer for throughput, without changing what it knows

*Approx. 20–25 minutes spoken. Narrated in the original Speechify voice, to match the earlier chapters. A companion to "The Model Is No Longer Just the Model": that lecture was about how models and serving systems are being designed together; this one assumes the model is already fixed and asks the narrower engineering question — how do we make exactly this thing run faster?*

---

## Cold open

Welcome back. Today's setup is simple. Somebody hands us a trained Transformer and tells us we are not allowed to change it. We cannot redesign its attention, we cannot retrain it, we cannot delete half its layers, we cannot turn it into a mixture of experts. The weights are the weights.

Our job is only this: make it execute faster.

That sounds like a narrow problem, and it is one of the richest problems in the field, because there are about six different levels at which we can attack it. At the very bottom we can optimize a single matrix multiplication. Above that, we can fuse operations together and move less data. Above that, we can rewrite attention and the key-value cache. Above that, we can change how requests get batched and scheduled. And above that, we can change how the model is spread across GPUs, and even put prefill and decoding on separate machines.

All of those levels interact, which means the first thing we need is not a kernel. It is a mental model of what we are actually optimizing.

---

## Part one — what "fast" even means

When someone says a model does one hundred tokens per second, that number is close to meaningless on its own, because there are three different things we might care about.

The first is time to first token. That is dominated by processing the user's prompt, which we call prefill.

The second is time per output token, which is dominated by the generation loop, which we call decode.

The third is total throughput — all the output tokens per second, summed across every user on the machine.

Here is the awkward part: these fight each other. If I have one request and I generate its next token immediately, latency is excellent and the GPU might be ten or twenty percent utilized. If I wait until sixty-four requests have arrived and process them together, GPU utilization and total throughput go way up, but every individual user waited longer.

This is why modern serving systems have mostly stopped chasing raw tokens per second and started optimizing what people call goodput — how much useful throughput you deliver while still meeting a latency target.

And the single most useful idea in this whole lecture is the one that falls out of this immediately: prefill and decode are not two phases of one workload. They are two different workloads that happen to share a set of weights. Systems like DistServe and Mooncake take that so seriously that they run them on separate machines.

---

## Part two — why decode is so much harder than it looks

Let us see why they are different, using the simplest possible layer: an output equals an input times a weight matrix.

During prefill, the input contains the whole prompt at once — maybe four thousand tokens. So we are multiplying a four-thousand-row matrix by a weight matrix. That is a big, fat matrix-matrix multiply. GPUs adore this. There is plenty of independent work, the weights get reused across thousands of rows, the Tensor Cores stay busy, and we get close to being limited by arithmetic — which is exactly where we want to be, because arithmetic is the thing the hardware is best at.

Now decode. Every active sequence contributes exactly one new token. With a single user, our input has one row. That beautiful matrix-matrix multiply has collapsed into something much closer to a matrix-vector multiply.

And notice what did not change: the entire weight matrix still has to participate. We still drag hundreds of megabytes, or gigabytes, of weights out of high-bandwidth memory — and then we do almost nothing with each number before moving on.

So decode is usually limited by memory traffic, not by arithmetic. That is the reason an H100 capable of an absurd number of operations per second can look embarrassingly idle while generating tokens for one user.

---

## Part three — arithmetic intensity, the one number to keep in your head

There is a single ratio that explains most of this. Arithmetic intensity asks: how much computation do I do for every byte I move?

Work through it roughly. The multiply-add count for our layer scales with the number of tokens times the two weight dimensions. The bytes we move, if weights dominate, scales with just the two weight dimensions. Divide one by the other, and both weight dimensions cancel.

What survives is the number of tokens.

That is a genuinely surprising result. The size of the weight matrix drops out. The thing that determines whether you are memory-bound or compute-bound is how many tokens you are pushing through at once.

One token in flight: intensity of about one, deeply memory-bound. One hundred and twenty-eight tokens in flight: every weight value gets reused a hundred and twenty-eight times, and intensity goes up by the same factor.

This is why batching is one of the most powerful optimizations ever applied to Transformers. We are not making the matrix multiply cleverer. We are changing the shape of the work into something the hardware likes.

And it is why quantization is so effective during low-batch decode. Take weights from sixteen bits down to four, and weight traffic drops roughly fourfold, so arithmetic intensity rises roughly fourfold. Quantization is not really "making the model smaller." It is moving the operation across the roofline, from bandwidth-limited toward compute-limited.

Every optimization decision in the rest of this lecture is really the question: which resource am I currently limited by, and am I attacking that one?

---

## Part four — inside the matrix multiply

Let us go one level down, into how a fast matrix multiply is actually built, because the same idea reappears at every level above it.

The naive version is three nested loops: for each row, for each column, walk the inner dimension and accumulate. Mathematically perfect, and a terrible way to feed a GPU — because the same values get fetched from main memory over and over.

So real kernels tile. A block of threads takes responsibility for, say, a one-hundred-and-twenty-eight by one-hundred-and-twenty-eight tile of the output. It does not do the whole inner dimension at once; it pulls in a narrow slice of each input, copies those slices from far-away global memory into fast on-chip shared memory, multiplies them, accumulates the partial result, then pulls the next slice.

So the flow is: main memory feeds shared-memory tiles, shared memory feeds per-warp tiles, those feed register fragments, and the fragments feed the Tensor Cores. The partial output sits in registers the entire time and only gets written back at the very end. That hierarchy is essentially what libraries like CUTLASS are built around.

And the reason it works is exactly arithmetic intensity again. One value fetched once and reused a hundred times instead of fetched a hundred times.

The obvious follow-up question is: why not use enormous tiles, and get enormous reuse? Because tiles cost resources. A bigger tile needs more shared memory and more registers for its accumulators. Push far enough and only one block fits on a processor at a time, occupancy collapses, and there is no longer enough independent work to hide stalls.

So tile size is a balance — enough area for reuse, not so much that you kill concurrency. That is why high-performance libraries ship dozens of kernels rather than one, and it matters enormously here: prefill produces huge token counts, decode produces tiny ones, and the same kernel is almost never right for both.

One more trick worth knowing, because it is the seed of everything in part seven. While the Tensor Cores chew on the current slice, we do not want the memory system sitting idle. So kernels use two buffers: one being consumed, one being filled. When compute finishes, they swap. That is double buffering, and instead of load, wait, compute, load, wait, compute, we get loading and computing happening on top of each other. On Hopper-class hardware, dedicated machinery for asynchronous copies and asynchronous Tensor Core operations lets this go further, with some warps acting purely as data movers and others purely as compute — a pattern called warp specialization.

---

## Part five — stop writing things down you are about to read back

A Transformer layer is not just matrix multiplies. There is normalization, the query-key-value projection, rotary embeddings, attention, the output projection, a residual add, another normalization, a gate projection, an up projection, an activation, a down projection, another residual.

A naive framework launches a separate GPU kernel for each of those. That costs us twice. Every launch has overhead, and — much worse — each kernel writes its output to main memory purely so the next kernel can immediately read it back.

Kernel fusion is the fix, and the mental rule is beautifully simple: every time you see an intermediate tensor, ask whether it actually needs to exist in memory. Very often it does not. A fused kernel does the matrix multiply and then, while the result is still sitting in registers, applies the bias, the activation, and the quantization before writing once. That trailing work is called the epilogue.

Transformers hand us obvious opportunities. Query, key and value all read the same input, so we can do one bigger projection instead of three separate ones. In the feed-forward block, the gate and up projections also read the same hidden state, so they merge too, and the activation and elementwise multiply fuse on afterwards. Normalization plus the residual update is another standard target.

And then there is the famous one.

---

## Part six — FlashAttention, and what it was really about

Standard attention multiplies queries by keys, takes a softmax, and multiplies by values. Conceptually there is an attention matrix whose size is sequence length by sequence length. At long context, that object is enormous, and traditional implementations wrote it to memory.

FlashAttention's insight was that this was never a compute problem. It was an input-output problem.

So instead of materializing that matrix, it tiles. A block of queries comes on chip. Blocks of keys and values stream past it. Running attention statistics accumulate, and an online softmax keeps the normalization correct as new blocks arrive. The answer is mathematically the same, and the giant intermediate never touches main memory at all.

What happened next is the most instructive part. FlashAttention-3 mapped that algorithm onto Hopper by leaning on asynchrony — overlapping Tensor Core work with data movement, splitting warps into producers and consumers, interleaving softmax with matrix multiplication — and reported roughly one-and-a-half to two times the speed of its predecessor.

Then FlashAttention-4 arrived on Blackwell and found something telling. Tensor Core arithmetic had gotten so much faster that the matrix multiplies were no longer the problem. The new bottlenecks were the exponential function inside softmax, and shared-memory traffic.

Watch the sequence there. First we optimized operations. Then main-memory traffic. Then shared memory. Then the special-function units that compute exponentials. Every time you kill a bottleneck, you promote the next one. That is not a failure of optimization. That is what optimization is.

---

## Part seven — the key-value cache is the real long-context story

Decode attention is its own beast. During prefill we have thousands of queries at once. During decode we have one query per sequence — but that one query has to attend over everything that came before it. Two thousand tokens. Thirty-two thousand. A hundred thousand.

So decode attention is another streaming problem: pull in the cache, compute the interactions, softmax, combine. And the amount of data grows with context length, which is why cache engineering matters so much.

The first big idea is PagedAttention. Suppose every user had to reserve a contiguous block of memory big enough for their longest possible sequence. We would waste an enormous amount, because sequences have wildly different lengths and finish at different times, and memory fragments.

PagedAttention treats cache memory the way an operating system treats virtual memory. A sequence's cache is chopped into fixed-size blocks, and those blocks do not need to be physically adjacent. A request logically owns blocks one, two, three, four, while physically those blocks are scattered across the pool. Near-zero waste, which means more sequences fit, which means bigger batches, which — back to part three — means higher arithmetic intensity. PagedAttention is not attention with fewer operations. It is memory virtualization that buys you batch size.

Block size is, once again, a tiling trade. Small blocks waste almost nothing when a sequence ends mid-block, but need more bookkeeping and give more scattered access. Large blocks are simpler and more local, but waste more at the tail. As always: tile size is a negotiation, never a direction.

The second idea is prefix caching, and it might be the highest-value item on this entire list for anything agentic. Imagine a thousand requests that share the same five-thousand-token system prompt. Why run prefill on those five thousand tokens a thousand times? Compute the cache once and reuse it. Automatic prefix caching does exactly this, and SGLang's RadixAttention generalizes it, storing reusable prefixes in a radix tree so overlapping — not just identical — prompts can share work. System prompts stay constant. Tool descriptions stay constant. Conversation histories overlap. The fastest token is the one you never compute.

The third idea is shrinking each cached element. Production stacks support eight-bit caches; research has pushed to two and three bits, with the nice finding from KIVI that keys and values want different treatment — keys quantized per channel, values per token.

But there is a systems lesson here that generalizes far beyond caches, so let me state it loudly: compression does not automatically mean acceleration. If I save four bytes and then burn a long sequence of instructions unpacking, rescaling and rearranging them, my theoretically superior format can be slower. The format and the kernel have to be designed together.

---

## Part eight — the same trap in weight quantization

That exact trap catches weight quantization too. Take sixteen-bit weights down to four bits and weight bandwidth drops fourfold. Wonderful. But the Tensor Cores cannot eat the packed representation directly. Something has to unpack the four-bit values, apply zero points, multiply by scales, convert formats, and rearrange the data before the arithmetic units see it. Do that clumsily and you hand back most of your gains.

This is why kernels like MARLIN matter — they treat low-bit inference as a kernel-scheduling problem rather than a numerical one, and hold onto close to the ideal bandwidth advantage across real batch sizes. So when you evaluate a quantization scheme for speed, the interesting question is not "how many bits." It is "what instructions actually execute between memory and the Tensor Core."

Which precision to pick follows straight from part three. At small batch, weight bandwidth dominates, so weight-only quantization is the win. As batch grows, weights get amortized across more tokens and arithmetic starts to dominate, so quantizing activations too begins to pay, because now you get genuinely lower-precision matrix multiplies. At long context, cache traffic can dominate everything, and you should be quantizing the cache instead.

There is no universally correct precision. Quantize whichever thing is currently responsible for the bytes or the operations that are limiting you.

---

## Part nine — scheduling is a hardware optimization in disguise

Back to that token count. One user gives us one row. Sixty-four users decoding at once give us sixty-four rows, and one fetch of a weight tile now serves all sixty-four. That is why throughput climbs so steeply with batch size, right up until some other resource saturates.

But naive batching wastes that. Take thirty-two requests: some finish after twenty tokens, some run five hundred. If we hold the whole batch until the slowest one finishes, we are burning capacity. So modern servers use continuous batching, sometimes called in-flight batching. At every single decode step, finished requests leave, waiting requests join, and the batch is rebuilt.

Chunked prefill is the companion trick. If one user submits a hundred-thousand-token prompt and we prefill it in one giant operation, every decode already in flight stalls, and their per-token latency explodes. So we cap how many prompt tokens are admitted per iteration and interleave prefill work with decode work. Larger budgets mean healthier matrix shapes and better throughput; too large and time-to-first-token suffers.

Notice what scheduling has become. By choosing how many tokens enter each iteration, we are directly choosing the shape of every matrix multiply in the model. Scheduling is a kernel optimization.

Then there is pure overhead, which becomes more embarrassing the faster your kernels get. Decode launches many tiny kernels thousands of times over, and the CPU has to submit each one. CUDA Graphs fix this by capturing a whole chunk of GPU work and replaying it as one unit. When NVIDIA did this for llama.cpp, they got up to about a one-point-two times improvement — biggest on small models, where launch overhead is the largest fraction of the runtime.

Think about what that means. No layers removed. No arithmetic reduced. They deleted the *nothing* in between the kernels, and it got faster.

The same logic explains why serving runtimes obsess over the host side: sampling on the CPU, Python scheduling, metadata rebuilds, memory allocation. The faster the GPU gets, the more the CPU becomes the problem — which is also why "GPU utilization" is a misleading metric. The GPU can be busy doing bad work, or idle because nobody is feeding it.

---

## Part ten — speculative decoding, and scaling outward

Everything so far still assumes one expensive model pass yields exactly one token. Speculative decoding attacks that assumption directly. A cheap draft mechanism proposes several likely next tokens, the big model verifies all of them in a single wider pass, and the ones that survive are kept. We have converted the pathological one-row workload into something with actual width.

But it is not free. If the draft is bad, we pay to generate candidates, the target rejects them, and we end up slower. What matters is average accepted length weighed against draft and verification cost. Medusa reported over two times speedups with lightweight prediction heads, and EAGLE-3 reports higher peaks still — but recent systematic evaluations are blunt that production-scale gains are much smaller than the headline numbers, and the documentation of major serving frameworks describes speculation as a low-batch optimization. That makes sense: at high concurrency, batching is already consuming the spare compute that speculation was hoping to borrow.

Scaling across GPUs has the same character. Tensor parallelism splits each weight matrix across devices, giving us more compute and more memory, but inserting a collective communication into essentially every layer. For big prefill multiplies there is enough arithmetic to hide that cost. For single-user decode it hurts: tiny multiply, synchronize, tiny multiply, synchronize. Sometimes a quantized model that fits on one GPU beats the same model split across four.

Pipeline parallelism instead gives different layers to different GPUs, which cuts the frequency of collectives but turns the model into a pipeline that only stays full if you have enough concurrent work. It is a throughput strategy, not a latency one.

And for mixture-of-experts models, the difficulty changes shape entirely. Routing sends different token counts to each expert — seventeen here, three there, forty-nine somewhere else, zero for that one — so instead of one large multiply you have many small irregular ones. Grouped matrix multiplication kernels handle a whole set of experts in one coordinated launch. And once experts live on different GPUs, tokens have to travel, so libraries like DeepEP exist specifically to make dispatch and combine cheap and to keep communication from stealing the processors that should be computing. The goal is that while one expert computes, the next expert's tokens are already arriving and the previous expert's results are already leaving.

The frontier here is fusing that entire chain — dispatch, first expert projection, activation, second projection, combine — into one overlapped pipeline. Not "make the multiply faster," but "schedule the whole layer as dataflow."

---

## Part eleven — what to actually do on Monday morning

If someone hands you a model tomorrow, do not start by writing a CUDA kernel. Start by measuring, in roughly this order.

Establish the workload first. Benchmark prefill and decode separately, across realistic prompt lengths, output lengths and concurrency levels. One tokens-per-second number is not a measurement.

Then make the model fit efficiently — choose a precision suited to your hardware and your accuracy budget, without assuming smallest equals fastest. Use the best available attention implementation for your GPU generation. Turn on paged cache management and continuous batching, which routinely deliver far more than hand-tuning any single kernel. Tune the chunked-prefill token budget until the matrix shapes are healthy but latency is still acceptable. Strip launch and framework overhead with graphs and memory pools. Only then start inspecting individual matrix shapes and using separate kernels for prefill and decode. Optimize the cache path — prefix reuse and cache precision — if your contexts are long. Try speculation if your concurrency is low enough for it to help. And only after all of that, scale outward.

When you do profile, look at the timeline before you look at any single kernel. Measure achieved memory bandwidth. Measure Tensor Core utilization. Look for gaps between kernels, stray memory copies, host stalls, collective traffic, expert imbalance. If a feed-forward multiply is already near the hardware limit, rewriting it is a waste of your life. If attention is sixty percent of decode time at long context, no amount of matrix-multiply tuning will save you. Optimize by wall-clock contribution, not by intellectual appeal.

---

## The one idea underneath all of it

Let me collapse this whole lecture into a single sentence, because everything we discussed is the same sentence wearing different hats.

**Increase the useful work done per expensive movement or synchronization.**

That is it. A tiled matrix multiply fetches data once and reuses it many times. Batching fetches weights once and uses them for many tokens. FlashAttention loads a block of keys and values and consumes it on chip instead of writing an attention matrix. Fusion keeps intermediates next to the arithmetic. Quantization moves fewer bytes per useful operation. Prefix caching refuses to recompute a prompt it has already seen. Speculative decoding extracts several tokens from one expensive verification. Expert-parallel overlap hides the network behind the math. CUDA Graphs collapse thousands of launches into one submission. Memory pools reuse buffers instead of rebuilding them.

Every single one is an attack on wasted movement or wasted waiting.

And that is why an unchanged Transformer — same weights, same architecture, same everything it knows — can end up dramatically faster than it was when someone handed it to you. You never made the model smarter. You just stopped making the hardware wait.
