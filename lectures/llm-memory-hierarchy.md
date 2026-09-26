# Memory Hierarchy for LLMs
*A narration-first lecture, roughly thirty minutes spoken.*

## Opening: why LLM performance is really about moving data

When people first learn about GPUs, they usually focus on computation. They hear about Tensor Cores, teraFLOPS, thousands of threads, and huge matrix multiplications.

But a GPU can only compute on data that has arrived.

A Tensor Core may be capable of an enormous number of operations per second, but if the next weights are still traveling from GPU memory, that Tensor Core can sit partially idle.

That gives us the central idea for this entire lecture:

**LLM performance is a data-movement problem as much as it is a compute problem.**

We are going to build one mental model connecting VRAM, HBM, cache, shared memory, registers, warps, Tensor Cores, tiling, quantization, attention, KV caches, and expert prefetching. By the end, phrases like "memory-bound" or "prefetch into cache" should correspond to a concrete hardware picture.

---

## Start with the complete memory journey

Imagine one weight from a transformer.

Before the model is loaded, that weight may be on an SSD.

When the program loads the model, data may pass through ordinary CPU system memory, or DRAM.

Then the model is transferred to GPU memory across an interconnect such as PCI Express.

Inside the GPU, the large main memory is what we commonly call VRAM. On an NVIDIA H one hundred SXM, that VRAM is HBM three, meaning High Bandwidth Memory.

During inference, a weight needed by a kernel is fetched through the GPU memory system. It may be served from the GPU's L two cache if it is already there, or from HBM if it is not.

Once data reaches a streaming multiprocessor, or SM, the kernel can make use of much smaller and faster on-chip storage.

That includes L one cache, explicitly managed shared memory, and registers belonging to individual threads.

Finally, the compute units execute instructions using those values.

So our simplified picture is:

SSD, then CPU RAM, then GPU HBM, then on-chip cache and shared storage, then registers, then arithmetic units.

The closer we get to the arithmetic units, the memory generally becomes faster but dramatically smaller.

That tradeoff is the reason the hierarchy exists.

---

## VRAM, HBM, DRAM, and SRAM

Let us clean up some terminology.

VRAM is a broad term. It simply means the main memory attached to the GPU.

On many consumer GPUs, the VRAM is built using GDDR memory.

On high-end accelerator GPUs such as the H one hundred SXM, the main GPU memory is HBM three.

HBM is still a form of dynamic random-access memory, or DRAM. It is designed to provide very high bandwidth by placing multiple memory stacks close to the GPU and using an extremely wide interface.

The H one hundred SXM, for example, has eighty gigabytes of HBM three and memory bandwidth measured in multiple terabytes per second.

That sounds unbelievably fast, and it is.

But from the perspective of a Tensor Core doing arithmetic on-chip, HBM is still far away.

Caches and shared memory are implemented using much faster on-chip memory structures, commonly described in terms of SRAM.

SRAM is fast, but physically expensive. It consumes much more chip area per stored bit than DRAM.

That is why we cannot simply put eighty gigabytes of SRAM beside the Tensor Cores.

Instead, we use a hierarchy.

Large DRAM holds the full model.

Small SRAM-like structures hold the tiny portion of the model that is useful right now.

The optimization objective is simple:

**Move useful data inward and reuse it before fetching more.**

---

## The GPU is made of streaming multiprocessors

Now let us move from memory to execution.

A modern NVIDIA GPU contains many streaming multiprocessors, abbreviated SMs.

Think of each SM as a local execution engine.

When you launch a GPU kernel, the total work is divided into thread blocks. The GPU schedules those thread blocks onto SMs.

Inside a thread block are many threads.

And on NVIDIA GPUs, threads are grouped into sets of thirty-two called warps.

A warp is one of the fundamental execution units you should picture when thinking about GPU software.

The thirty-two threads in a warp execute the same kernel instruction stream under NVIDIA's SIMT model, meaning single instruction, multiple threads.

Each thread has its own data and its own state.

This structure is ideal for matrix operations. Instead of asking one CPU-like thread to compute a giant multiplication, we divide the matrices into tiles. Different blocks handle different output tiles, warps cooperate inside the blocks, and individual threads contribute to smaller fragments.

So picture the execution hierarchy as GPU, then SM, then thread block, then warp, then thread. Those threads ultimately cooperate to feed the arithmetic units.

---

## Registers — each thread's private working area

The closest general-purpose storage to a thread is its registers.

Imagine every thread has a tiny personal workbench.

A thread can keep loop variables, addresses, intermediate values, fragments of input data, and partial sums in registers.

For LLM matrix multiplication, partial sums are extremely important.

Suppose we are computing part of an output matrix.

We do not want to perform one multiplication, write the partial answer all the way back to HBM, load it again, add another multiplication, and repeat.

That would create enormous memory traffic.

Instead, the kernel keeps partial accumulations close to the compute units while many multiply-and-accumulate operations happen.

Registers are ideal for this because access is extremely fast.

But there is a tradeoff.

The register file on an SM is finite.

If each thread demands too many registers, fewer warps can live on that SM at the same time.

This is called register pressure.

And if the compiler cannot keep everything in registers, some values may spill into what CUDA calls local memory.

Despite the word local, spilled local memory is not another tiny register-like store. It is backed by device memory and uses the cache hierarchy, so heavy spilling can be expensive.

So the lesson is:

**Registers are extremely valuable, but they are a limited resource that must be budgeted.**

---

## Why we want many warps: latency hiding

Suppose one warp requests data and has to wait.

A CPU might spend substantial effort trying to make one thread execute as quickly as possible.

A GPU uses a different strategy.

It keeps many warps available.

If warp A is waiting on memory, the SM can issue instructions from warp B.

Then perhaps warp C.

Then warp D.

By the time the scheduler returns to warp A, its data may have arrived.

This is called latency hiding.

It explains why occupancy matters.

Occupancy roughly describes how many warps are resident and available on an SM relative to the hardware limit.

However, maximum occupancy is not automatically maximum performance.

A high-performance matrix multiplication may intentionally use many registers for accumulator fragments and a substantial amount of shared memory for tiles.

That can reduce the number of blocks or warps that fit simultaneously.

But the kernel may still run faster because each active warp performs much more efficient work.

So do not memorize:

"High occupancy equals good."

Memorize:

**We want enough parallel work to hide latency, but we also want enough registers and shared memory to maximize reuse.**

GPU kernel optimization is often about finding that balance.

---

## Shared memory — the block's cooperative scratchpad

Registers are private to individual threads.

Shared memory is available to the threads of a block.

Think of registers as personal notebooks and shared memory as a whiteboard used by the entire team.

Suppose a block of threads needs the same tile of a weight matrix.

A bad design might repeatedly fetch overlapping data from farther out in the memory hierarchy.

A better design loads the tile once into shared memory and allows many threads to reuse it.

That is one of the central ideas behind tiled matrix multiplication.

There is an important nuance here.

Earlier, we can describe the hierarchy as HBM, then L two, then shared memory, then registers.

That is useful conceptually, but shared memory is not simply another automatic cache.

L two and L one are hardware-managed caches.

Shared memory is explicitly managed by the kernel.

The program deliberately allocates it, deliberately loads values into it, and deliberately decides how threads will reuse those values.

On H one hundred, L one cache, texture cache, and shared memory use a combined on-chip resource. The exact partition can be configured, with more than two hundred kilobytes per SM available for shared memory in supported configurations.

So shared memory gives the programmer direct control over a small, fast part of the memory system.

That control is one of the main tools used by optimized LLM kernels.

---

## Coalescing and bank conflicts

Fast memory is not enough.

You must also access it correctly.

When the thirty-two threads of a warp load from global memory, the hardware tries to combine their requests into a small number of memory transactions.

This is called coalescing.

If neighboring threads request nearby addresses, the GPU can usually make efficient use of each memory transaction.

If the threads request scattered addresses, the hardware may need many more transactions.

That wastes bandwidth.

Now move into shared memory.

Shared memory is divided into banks so multiple accesses can happen in parallel.

If different threads access addresses that map nicely across those banks, the accesses are efficient.

But if many threads access different addresses that map to the same bank, the requests may become serialized.

That is called a bank conflict.

So when optimizing an LLM kernel, there are two different questions.

First:

Did I move the data into fast memory?

Second:

Did I arrange the data so the threads can access it efficiently?

This is why memory layout matters so much.

Two kernels can implement the exact same mathematical matrix multiplication and have radically different performance because one produces coalesced global loads and conflict-free shared-memory access while the other does not.

---

## Tensor Cores — the arithmetic engines

Now we reach the Tensor Cores.

Tensor Cores are specialized hardware units designed to perform matrix multiply-and-accumulate operations at extremely high throughput.

Conceptually, the operation is:

C becomes A multiplied by B, plus C.

A large LLM matrix multiplication is decomposed into many smaller matrix operations that ultimately feed these units.

Modern Tensor Cores support multiple numerical formats. Hopper-generation GPUs support formats especially useful for transformers, including BF sixteen, FP sixteen, and FP eight paths.

But Tensor Cores do not magically solve memory problems.

They make the arithmetic side extremely fast.

That can actually make memory limitations more important.

Imagine a hypothetical kernel where moving the data takes ten units of time and computing on it takes two units.

Now suppose we double Tensor Core speed.

The compute portion falls from two units to one.

But the ten units spent moving data remain.

Total time only changes from twelve units to eleven.

The workload barely speeds up.

This is the essence of a memory-bound operation.

So when someone quotes the peak Tensor Core performance of a GPU, remember that peak compute only matters if the rest of the system can feed it.

---

## Tiling: how a GEMM uses the hierarchy

Consider a transformer operation where activation matrix X is multiplied by weight matrix W.

The complete matrices are far too large to fit in registers or shared memory.

So the kernel divides them into tiles.

A thread block takes responsibility for a tile of the output.

To compute that tile, it needs corresponding pieces of X and W.

Those pieces are loaded from global GPU memory.

If the requested data is already in L two cache, some requests can be served there instead of going all the way to HBM.

The kernel then stages reusable pieces into on-chip storage, commonly shared memory.

Warps consume smaller fragments from those tiles.

Tensor Core instructions perform multiply-and-accumulate operations.

Partial output values stay close to the arithmetic units, often in registers or architecture-specific accumulator structures.

Then the kernel moves along the reduction dimension.

Load the next tile.

Multiply.

Accumulate.

Load the next tile.

Multiply.

Accumulate.

Eventually, the final output tile is written back to global memory.

We cannot eliminate memory movement. The goal is to make every expensive load support as much useful arithmetic as possible. Tiling is therefore fundamentally a data-reuse strategy.

---

## Pipeline the loads instead of waiting

A naive kernel might perform these steps:

Load a tile.

Wait for it.

Compute.

Load the next tile.

Wait again.

Compute again.

That creates idle gaps.

A more sophisticated kernel overlaps movement with computation.

While Tensor Cores are operating on the current tile, the next tile begins moving toward the place where it will be needed.

This idea is often implemented using pipelining or double buffering.

One buffer is actively being consumed.

Another is being filled with future data.

When the current computation finishes, the next data is already available or much closer to ready.

Hopper GPUs also contain mechanisms such as the Tensor Memory Accelerator, or TMA, designed to help transfer multidimensional data between global and shared memory asynchronously.

The general principle matters more than the particular instruction:

**Do not merely reduce data movement. Hide the unavoidable movement behind useful work whenever possible.**

That principle will become very important when we discuss mixture-of-experts prefetching.

---

## Prefill versus decode

This is one of the most important distinctions in LLM inference.

During **prefill**, the model processes the prompt.

If the prompt contains hundreds or thousands of tokens, the GPU can process many token activations together.

That creates large matrix multiplications.

A weight loaded from HBM can contribute to the computation of many tokens.

So each byte of weight data supports a large amount of arithmetic.

We call that high arithmetic intensity.

During **decode**, the model usually generates one new token at a time for each active sequence.

The weight matrices remain enormous.

But now the amount of activation work for one sequence is much smaller.

The GPU may stream a huge quantity of weights through the memory system to compute just one new token.

There is much less weight reuse per sequence.

That makes decode much more likely to become memory-bandwidth bound.

This is why the same GPU can behave very differently during prefill and decode.

Prefill often has large GEMMs that use Tensor Cores efficiently.

Single-sequence decode often has much skinnier operations and a much stronger dependence on memory bandwidth.

Whenever you hear that an optimization improves "LLM inference," ask:

**Does it improve prefill, decode, or both?**

That question prevents a lot of misleading conclusions.

---

## Arithmetic intensity and the roofline idea

There is a useful concept called arithmetic intensity.

The definition is simple:

**Arithmetic intensity is the amount of useful computation performed per byte of data moved.**

If you load a large weight tensor and use each value only once, arithmetic intensity is low.

If you load that same weight and reuse it across many tokens, arithmetic intensity rises.

This connects to the roofline model.

Imagine two ceilings on performance.

One ceiling comes from compute throughput.

The other comes from memory bandwidth.

At low arithmetic intensity, memory bandwidth usually limits performance first.

At high arithmetic intensity, the kernel may eventually become compute-bound.

This gives us a clean explanation for batching.

Suppose one decode request generates one token.

The model weights are loaded for that work.

Now suppose we batch many decode requests together.

The same weight tile can contribute to multiple tokens.

We perform more arithmetic for roughly the same weight load.

Arithmetic intensity rises.

Tensor Core utilization can improve.

So batching is not merely a serving-system trick.

It changes the hardware economics of the matrix multiplication.

---

## Why quantization can speed up decode

Now quantization becomes intuitive.

Suppose model weights are stored using sixteen bits per value.

If we represent them using four bits, the raw weight storage becomes roughly one quarter as large, ignoring scales, metadata, padding, and other overhead.

If decode is limited by moving weight bytes from HBM, then reducing the number of bits that must be transported creates a direct opportunity for speedup.

But there is a catch.

The compute kernel has to consume the compact representation efficiently.

Imagine a bad one-bit or four-bit implementation.

The packed weights are loaded.

They are unpacked.

They are converted to a wider representation.

The expanded weights are written into a large temporary buffer in HBM.

Then an ordinary matrix multiplication reads them again.

The model may be small on disk and small when first loaded, but the execution path has recreated a large memory-traffic problem.

A better implementation keeps the weights compressed for as long as possible.

Small tiles are unpacked close to the computation.

Scaling and conversion are fused into the kernel.

Expanded values exist briefly in registers or shared memory instead of being materialized as a giant global tensor.

This is why quantization formats and kernel design cannot be separated.

**Compression only produces speed when it reduces bytes moved along the actual execution path.**

---

## The KV cache is another bandwidth problem

Weights are only part of the memory story.

Autoregressive attention maintains a key-value cache, usually called the KV cache.

For previous tokens, the model stores key and value vectors that future tokens may attend to.

As the context becomes longer, the cache grows.

So there are two separate issues.

The first is capacity.

How much KV data can fit in GPU memory?

The second is bandwidth.

How much historical KV data must be read when computing the next token?

Grouped-query attention helps by letting several query heads share a smaller number of key and value heads.

KV-cache quantization can reduce the number of bytes stored and transferred.

Paged-attention systems organize cache memory into manageable blocks so that serving many variable-length sequences does not require one enormous contiguous region for every request.

Decode has several memory streams at once: weights, KV data, activations, and temporaries. Removing one bottleneck can expose another, which is why profiling matters.

---

## FlashAttention: same math, smarter movement

FlashAttention is one of the clearest examples of memory-aware algorithm design.

Standard attention conceptually computes a matrix of similarities between queries and keys.

For long sequences, that attention matrix can become enormous.

A naive implementation may write large intermediate matrices to HBM and later read them back.

The mathematics is correct, but the data movement is expensive.

FlashAttention reorganizes the calculation into tiles.

Blocks of queries, keys, and values are processed using fast on-chip memory.

An online softmax procedure allows the algorithm to produce exact attention without storing the entire attention-score matrix in HBM.

The important point is that FlashAttention does not primarily win by changing the definition of attention.

It wins by changing how the computation interacts with the memory hierarchy.

The original work explicitly describes this as IO-aware attention.

This teaches a general optimization lesson:

**Sometimes the biggest speedup comes from keeping the same mathematics but changing where intermediate values live.**

Whenever you see a huge intermediate tensor, ask:

Does this really need to be written to HBM?

Or can it be consumed while it is still on-chip?

---

## Kernel fusion follows the same principle

Suppose operation A creates an intermediate tensor.

Operation B immediately consumes it.

A naive execution might run operation A, write the complete intermediate tensor to HBM, launch operation B, and then read the tensor back.

That round trip can be expensive.

If the operations can be fused, intermediate values may stay in registers or shared memory long enough to be consumed directly.

That reduces global-memory traffic.

LLM implementations use fusion in many places.

An activation function can be fused with nearby linear operations.

Normalization can sometimes be combined with neighboring work.

Quantization or dequantization can be integrated into the kernel that consumes the values.

But fusion is not automatically good.

A larger kernel may use more registers.

It may use more shared memory.

It may reduce occupancy.

So the correct question is not:

"Can these operations be fused?"

The better question is:

**Does the fusion eliminate enough expensive memory traffic to justify the extra on-chip resource pressure?**

Again, everything comes back to balancing the hierarchy.

---

## Mixture of experts and prefetching

Mixture-of-experts models create an especially interesting memory problem.

A router chooses only a small subset of experts for each token.

For example, a model might contain many experts while activating only a handful per token.

That saves arithmetic because we do not execute every expert.

But expert access becomes irregular.

Different tokens may select different expert weights.

Different layers may select different experts.

This creates a natural opportunity for prediction.

Suppose information available in the current layer lets us predict which experts are likely to be used in the next layer.

We could begin memory movement before the exact demand occurs.

The point is not necessarily to force the entire expert into the smallest possible memory.

The point is to move useful data closer to where it will be needed, or at least start the request early enough that memory latency overlaps current computation.

For example, a prediction could help increase the chance that useful expert data is already present in L two when the next layer requests it.

But a prefetch can also hurt.

Wrong experts consume bandwidth.

Too many candidate experts consume cache capacity.

Prefetch too early and the data may be evicted.

Prefetch too late and the transfer is still exposed on the critical path.

So prediction accuracy is not the final metric.

The real question is:

**How much correct memory latency did the prediction successfully hide?**

That is the hardware-level objective.

---

## L two cache versus shared memory in prefetching

This distinction is worth making explicit.

L two is a large cache shared by the GPU's SMs.

On an H one hundred, it is around fifty megabytes.

Shared memory is much smaller and belongs to a thread block on an SM.

You usually do not manage ordinary L two cache lines in exactly the same explicit way that you allocate shared memory.

So when someone says, "Prefetch the expert into cache," ask which stage they actually mean.

They may mean:

Start the HBM request earlier so the memory system is already working.

Or:

Increase the probability that the relevant data is resident in L two before the expert kernel begins.

Or:

Once the kernel starts, asynchronously stage the next tile from global memory into shared memory while Tensor Cores process the current tile.

These are related ideas, but they happen at different levels.

A clean framework is:

**Prediction tells us what data we expect to need.**

**Prefetch scheduling tells us when to begin moving it.**

**The memory hierarchy tells us how close to computation we can place it before use.**

That distinction is essential when designing a mixture-of-experts prefetch experiment.

---

## One complete decode-token walkthrough

Let us trace one token through a transformer.

The model weights are already stored in GPU HBM.

A token representation enters a transformer layer.

Normalization produces an activation vector.

Linear projections require large weight matrices.

Tiles of those weights move through the memory hierarchy toward the SMs.

Warps cooperate on the matrix operations.

Shared memory may stage reusable tiles.

Registers hold thread-local fragments and partial accumulations.

Tensor Cores execute multiply-and-accumulate operations.

Attention then combines the new query with historical keys from the KV cache.

Historical values contribute to the attention output.

A memory-efficient attention kernel tries to minimize unnecessary HBM reads and writes.

Then comes another projection.

Next is the feed-forward network.

In a dense transformer, more large weight matrices are streamed and multiplied.

In a mixture-of-experts transformer, a router first selects experts, and the selected expert weights are used.

The result becomes the input to the next transformer layer.

This repeats layer after layer.

Finally, the last hidden state is transformed into vocabulary logits and the next token is selected.

Then the entire process repeats for the following token.

So the true story of inference is not:

"Tensor Cores multiply matrices."

The fuller story is:

**The system repeatedly predicts, moves, stages, reuses, computes, synchronizes, and stores data while trying to keep expensive off-chip traffic off the critical path.**

---

## Why a giant GPU can still struggle with one sequence

An H one hundred has enormous parallel compute capability.

So why can single-sequence token generation still fail to use all of it?

Autoregressive decoding contains a sequential dependency.

Token one hundred and one depends on the result of token one hundred.

Within one token there is substantial parallelism, but we cannot arbitrarily compute many future tokens at the same time with ordinary decoding.

That means one sequence can produce relatively skinny matrix multiplications.

The GPU may have far more arithmetic capacity than one token can efficiently use.

Serving systems therefore use continuous batching.

They combine work from many active sequences.

Now one weight tile can support computations for several current tokens.

The GPU performs more arithmetic for the same weight movement.

Tensor Core utilization improves.

This shows how tightly the software and hardware stacks are connected. A scheduler changes batch shape; batch shape changes GEMM dimensions; GEMM dimensions change weight reuse and arithmetic intensity; and that changes whether the kernel is limited by compute or memory. High-performance LLM inference is hardware-software co-design even without a custom chip.

---

## A practical framework for analyzing any LLM kernel

When you encounter a new LLM optimization, ask five questions.

First:

**What data does the operation need?**

Weights, activations, KV-cache blocks, routing scores, expert weights, or something else?

Second:

**Where does that data live before it is used?**

HBM, L two, another GPU, or CPU memory?

Third:

**How many times can each byte be reused once it reaches the chip?**

This tells you a lot about arithmetic intensity.

Fourth:

**Where do intermediate values live?**

Registers, shared memory, or HBM?

Could a fusion or tiled algorithm avoid writing them out?

Fifth:

**What resource is actually limiting performance?**

HBM bandwidth?

Tensor Core throughput?

L two behavior?

Shared-memory bandwidth?

Register pressure?

Inter-GPU communication?

Kernel-launch overhead?

If you can answer those five questions, you can usually explain why an optimization works.

---

## Final recap

Let us compress the lecture into one final mental model.

The model lives in large memory, while compute operates on small pieces. The GPU therefore keeps pulling useful tiles inward.

HBM stores the large working set.

L two can catch useful data shared across the GPU.

L one and shared memory provide fast on-chip access within an SM, with shared memory explicitly controlled by the kernel.

Registers hold the immediate state of threads and partial results.

Warps organize thirty-two threads.

Thread blocks let those threads cooperate.

SMs schedule their work.

Tensor Cores perform dense matrix arithmetic at enormous speed.

High-performance kernels tile large tensors, coalesce memory accesses, avoid shared-memory bank conflicts, reuse data on-chip, overlap future loads with current compute, and avoid writing unnecessary intermediate tensors back to HBM.

For LLMs, this explains almost everything we have discussed.

Prefill is often efficient because many tokens reuse the same weights.

Decode is often memory-bound because relatively little computation is performed for each large stream of weights.

Quantization can accelerate decode because fewer weight bits have to move.

FlashAttention accelerates attention by reducing costly HBM traffic.

Kernel fusion keeps temporary data close to compute.

KV-cache optimizations reduce capacity and bandwidth pressure.

Batching increases reuse.

And mixture-of-experts prefetching attempts to begin future memory movement before it becomes a blocking demand.

The most important sentence to remember is:

**A GPU is fastest when expensive data movement is minimized, useful data is reused close to the compute units, and unavoidable movement is overlapped with useful work.**

That turns the memory hierarchy from a list of hardware terms into one system, and it explains why two implementations of the same LLM can have dramatically different latency and throughput.
