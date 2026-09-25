# Compression Is Part of the Compute

### Lecture 1a: why a quantized model is only as fast as the kernel that reads it

*Approx. 45 minutes spoken. Written to be listened to: the numbers that matter are rounded and said in words, and the mechanisms are described rather than drawn. The one-line version: optimise the representation the hardware actually executes, not just the size of the file.*

---

## The obvious story

Welcome back. Today's lecture starts with an idea so obvious that it barely seems worth a lecture, and then spends forty minutes showing why the obvious version is incomplete.

Here is the obvious idea. A large language model is mostly weights. When the model generates a token, it has to read those weights from memory. Reading memory is slow. So if we store each weight in fewer bits, we move fewer bytes, and the model runs faster. Sixteen bits down to four bits means a quarter of the bytes, which should mean something close to four times the speed. Three bits should be faster still. Two bits faster again.

That story is not wrong. It is where the whole field of weight quantization for inference comes from, and the best systems really do get close to that four-times figure in the right conditions. But the story skips a step. Between the compressed bytes sitting in memory and the multiply-add that actually uses them, something has to unpack those bits into a number the arithmetic units understand. That unpacking is work. It happens on the same chip, in the same time budget, competing for the same resources as everything else. And depending on how the bits are laid out, it can be nearly free, or it can eat the entire benefit of compression, or it can make the compressed model slower than the uncompressed one.

So the thesis of this lecture is a single sentence. A compressed model is only fast if the processor can consume that compressed representation efficiently. Compression is part of the compute.

To make that sentence mean something, we need to build up some machinery. We will start with where the time actually goes when a model generates text. Then we will look at what fewer bits buy on paper, and what the hidden decoding step costs. Then we will go down into how a graphics processor physically reads memory, how its matrix units want their data arranged, and why lookup tables are awkward. Only then will we arrive at FLUTE, a kernel for lookup-table quantized models, and take it apart piece by piece. After that we will look at the same idea appearing in other systems, at real cases where a smaller format ran slower, at hardware that is starting to speak these formats natively, and finally at a design rule for anyone building their own hardware or kernels.

## Where the time goes

Let's start with the physics of generating one token.

A transformer layer is dominated by a few large matrix multiplications: the attention projections and the feed-forward network. When the model is generating text for a single user, one token at a time, each of those multiplications takes a single activation vector and multiplies it by a large weight matrix. That is a matrix-vector product, and it has a very particular shape of cost.

Think about what a matrix-vector product does per weight. It reads the weight, multiplies it by one element of the input vector, and adds the result into a running sum. Two floating-point operations, a multiply and an add. And the weight, in half precision, is two bytes. So the arithmetic intensity, the number of operations per byte fetched from memory, is about one. One operation per byte.

Now compare that to what the hardware can do. A modern data-centre graphics processor like the H100 can perform nearly a thousand trillion half-precision operations per second on its tensor cores, and it can read a little over three trillion bytes per second from its high-bandwidth memory. Divide one by the other and you get the processor's balance point: about three hundred operations per byte. On the older A100, it is about a hundred and fifty.

This is the roofline model, and it is worth holding onto because the rest of the lecture hangs from it. The roofline says the speed of any computation is capped by the lower of two ceilings: the processor's peak arithmetic rate, or its memory bandwidth multiplied by the computation's arithmetic intensity. If your work does fewer operations per byte than the balance point, you are memory bound: the arithmetic units sit idle, waiting for data. If it does more, you are compute bound.

Single-user decoding sits at about one operation per byte, against a balance point of about three hundred. It is not a little memory bound. It is memory bound by more than two orders of magnitude. The tensor cores spend almost all their time waiting.

That gives us a very simple way to estimate the speed limit of generation. Every token requires streaming essentially all the weights once. So the maximum tokens per second is roughly the memory bandwidth divided by the size of the model in bytes. An eight-billion-parameter model in half precision is about sixteen gigabytes. At a little over three terabytes per second, that is a ceiling of about two hundred tokens per second. Store the same model in four bits, around four gigabytes, and the ceiling rises to about eight hundred. Real systems reach perhaps half to two thirds of peak bandwidth, but the proportion holds: the bytes are the budget.

There is an energy version of the same argument. Fetching a word from off-chip memory costs hundreds of times more energy than a floating-point addition. Arithmetic is cheap. Moving data is expensive. So in this regime, bytes are the currency, and quantization is a way of spending fewer of them.

That is the obvious story, stated carefully. Now let's see what it hides.

## What fewer bits buy on paper

The first thing the obvious story hides is that a four-bit model is rarely four bits.

To store weights in four bits, you have to map real-valued weights onto sixteen possible codes. The standard way is group-wise quantization: take a small group of consecutive weights, say a hundred and twenty-eight of them, find a scale for that group, and store each weight as a small integer that gets multiplied by the scale to reconstruct it. Often there is also a zero point, an offset. The scale and zero point are stored in higher precision, typically sixteen bits each.

That metadata is not free. A sixteen-bit scale shared across a hundred and twenty-eight weights adds an eighth of a bit per weight. Add a zero point and it is a quarter of a bit. Use smaller groups for better accuracy and the overhead grows. The popular format for bits-and-bytes normal-float four-bit, with a scale for every sixty-four weights, comes to four and a half bits per weight, unless you also quantize the scales themselves, which brings it back to just over four point one. The widely used block formats in the llama dot c-p-p ecosystem tell the same story in their names: their "four-bit" format is four and a half bits per weight, their "three-bit" is about three and a half, and their "two-bit" is about two and a half.

Those overheads show up directly in speedups. The Marlin kernel, which we will meet properly later, is one of the best four-bit kernels ever written, and its authors point out that the ideal speedup over half precision is not four, but about three point nine, precisely because the group scales add an eighth of a bit to every weight. Metadata eats into the ceiling before a single instruction runs.

And for exotic formats, the bookkeeping gets worse. Methods that keep a small fraction of outlier weights in full precision, stored as a sparse matrix, pay for the indices of those outliers. One such method, SqueezeLLM, moves from about three point zero two bits to about three point two four bits by keeping less than half a percent of weights as outliers, and adds around ten percent to latency for handling them. Methods based on codebooks pay for the codebooks. A so-called two-bit model can easily be closer to three bits per parameter once you count everything in the file.

So the first correction to the obvious story is this: count every byte. The number that matters is bytes moved per weight, including scales, zero points, codebooks, outlier indices, and padding. Not the nominal bit width on the label.

## The hidden step: someone has to decode

The second thing the obvious story hides is the decoding step.

Most weight-only quantization schemes work like this. The weights are stored compressed. The activations stay in sixteen-bit floating point. The matrix multiplication itself happens in sixteen-bit floating point on the tensor cores, because that is what the tensor cores are built to do. So somewhere between memory and the tensor core, each compressed weight has to be turned back into a sixteen-bit number. Unpack the bits, look up or compute the value, multiply by the group scale, maybe add the zero point.

Where does that happen? The naive answer is: in a separate step. Launch one kernel that reads the compressed weights and writes out a full sixteen-bit copy, then launch an ordinary matrix multiplication on that copy. This is catastrophic for decoding, and it is worth seeing why. The separate step reads the compressed weights, which is the small read we wanted, but then writes the full sixteen-bit matrix back to memory and reads it again for the multiplication. You have made the traffic worse than not quantizing at all. One measurement of a four-bit implementation that dequantized in a separate kernel found it running at under a third of the speed of the plain half-precision baseline. An early version of a popular four-bit library only used a fused path for single tokens; for anything larger it expanded the whole weight to sixteen bits first.

So the decode has to happen inside the matrix multiplication kernel: load compressed bytes from memory, unpack them in registers, feed them straight to the tensor cores, and never write the expanded weights anywhere. That is called a fused, or mixed-input, kernel.

But fusing the decode does not make it free. It moves it onto a different part of the chip. The tensor cores do the matrix arithmetic. The unpacking, the shifts and masks and conversions and scale multiplications, happens on the ordinary arithmetic units, sometimes called the CUDA cores. And those are much, much slower. On an H100, the general-purpose floating-point throughput is about one fifteenth of the half-precision tensor core throughput. Look closer, at a single streaming multiprocessor per clock cycle, and the gap is starker. The tensor cores can do on the order of two thousand half-precision multiply-adds per cycle. The integer units can do sixty-four simple operations. And the native instruction that converts an integer into a floating-point number runs at just sixteen per cycle.

Now put that next to the memory budget. If the kernel is streaming weights at full memory bandwidth, how many instructions can it afford to spend decoding each weight without becoming the bottleneck itself? The rough answer, on an H100 at batch size one, is about two integer operations per weight, and less than one of those native conversions. Two instructions. That is the whole decode budget. Anything that costs more than a couple of cheap operations per weight turns a memory-bound kernel into an instruction-bound one, and you lose the speedup you paid for with accuracy.

This is why kernel writers treat the integer-to-float conversion with such care. The standard trick, which Marlin uses, avoids the slow conversion instruction entirely. You take the four-bit integer and, with a single logical instruction, drop it into the low bits of a sixteen-bit floating-point number whose exponent is pre-set so that the number represents one thousand and twenty-four plus your integer. Then you subtract one thousand and twenty-four, and you have your value as a half-precision float. A logical operation and a subtraction, done on two values at once because they are packed into one thirty-two-bit register. That kind of bit-level trick is not a curiosity. It is what makes four-bit decoding fit into the budget at all.

So here is the second correction to the obvious story. The format is not just a storage decision. It determines the decode instruction sequence, and that sequence has a budget of roughly two cheap operations per weight. A format that needs more than that is not faster, no matter how few bits it uses.

## How the processor actually reads memory

The third thing the obvious story hides is that memory is not read one bit, or one byte, at a time. It is read in fixed-size, aligned chunks, and the shape of those chunks matters enormously.

On NVIDIA graphics processors, global memory is accessed in thirty-two-byte sectors, grouped into hundred-and-twenty-eight-byte cache lines. A warp, which is a group of thirty-two threads that execute in lockstep, issues a memory instruction, and the hardware combines the addresses those threads asked for into the minimum number of thirty-two-byte transactions. If the thirty-two threads read thirty-two consecutive, aligned words, that collapses into a handful of transactions, and every byte fetched is used. That is called coalescing, and it is the first rule of fast kernels. If each thread reads from a scattered location, each request may cost a full sector, and you can end up using an eighth of the bandwidth you are paying for. Even a misaligned but otherwise sequential read can touch five sectors instead of four.

There is a second rule. An individual load instruction moves one, two, four, eight, or sixteen bytes, and the address has to be naturally aligned to that size. The fastest kernels use the sixteen-byte loads, a hundred and twenty-eight bits at a time, so that each thread pulls in a meaningful chunk per instruction.

Now think about what this means for bit widths.

Eight bits: a byte. Perfect. Four bits: two per byte, eight per thirty-two-bit word, thirty-two per sixteen-byte load. Perfect. Two bits and one bit: also powers of two, also perfect.

Three bits: awkward. Three does not divide eight. Pack three-bit values end to end, and some of them straddle byte boundaries, and some of them straddle word boundaries. Sixteen of them occupy forty-eight bits, which is six bytes, which is not a legal load size. To extract a particular value, a thread may need bits from two different words, which means extra loads, shifts, masks and ors, all of which come out of that two-instruction budget. The alternative, padding each three-bit value out to four bits so that everything aligns, wastes a quarter of the bandwidth you compressed to save.

Six bits has the same problem. Sixteen six-bit values are twelve bytes, again not a legal load size. One team measured that naive six-bit reads wasted between sixty and eighty percent of the shared-memory bandwidth involved.

So the third correction is this: the hardware has a native grain. Bit widths that align with it are cheap to load. Bit widths that don't are either expensive to unpack or wasteful to pad. The number three looks like it is between two and four. To a memory system, it is not between them at all. It is off to the side.

## Registers and the shape the tensor core wants

The fourth thing hidden in the obvious story is that getting the bytes onto the chip is not the end. They have to arrive in the right places.

Tensor cores do not multiply arbitrary arrays. A tensor-core instruction multiplies small fixed-size tiles, for instance a sixteen by sixteen tile against a sixteen by eight tile, and it expects its operands to be spread across the registers of the thirty-two threads in a warp in a very specific pattern, called the fragment layout. Thread zero holds certain elements, thread one holds certain others, and so on, in an interleaved pattern that exists for the convenience of the hardware, not for the convenience of anyone storing a model.

When you multiply ordinary sixteen-bit matrices, there are special instructions that load tiles from shared memory and scatter them into exactly this layout for you. But there is no such instruction for four-bit data, let alone three-bit data. So if your compressed weights are stored in the natural row-by-row order of the original matrix, every thread has to load bytes, unpack them, and then shuffle values between threads or registers to get them into the fragment layout. More instructions, from the same tiny budget.

The answer, used by every serious mixed-input kernel, is to do that shuffle once, offline, when the model is prepared. You permute the compressed weights in memory so that when each thread performs its single sixteen-byte load, the bytes it receives contain exactly the weights it will need, in exactly the order the tensor core expects. Marlin does this: each thread's sixteen-byte load holds exactly its own eight weights, already arranged in the interleaved order its registers need, so decoding is a handful of bit operations per pair and nothing moves between threads. On newer Hopper processors, whose matrix instructions read one operand directly from shared memory rather than registers, a follow-on kernel called Machete had to derive an entirely new pre-packed layout from the new instruction's requirements, and swap which operand lives where, so that the dequantized weights could stay in registers.

This is the first time we see the thesis in its full form. The on-disk order of the weights is not the model's natural order. It is an order chosen to match the memory transaction size, the thread count of a warp, and the fragment layout of a particular tensor-core instruction on a particular generation of hardware. The format, the layout, and the kernel are one design.

## Non-uniform formats and the lookup problem

So far we have talked about uniform quantization: codes are evenly spaced integers, and decoding is a scale and an offset. But evenly spaced levels are not the best use of a small number of bits. Neural network weights are roughly bell-shaped: most weights are near zero, a few are large. If you only get sixteen levels, you would rather spend more of them near zero where the weights are dense, and fewer out in the tails.

That is the idea behind non-uniform formats. The best known is normal-float four-bit, introduced with QLoRA, whose sixteen levels are placed at quantiles of a normal distribution, so that each level covers an equal share of a bell-shaped weight distribution. Other methods go further and learn the levels, with clustering, or with codebooks that represent several weights at once.

Non-uniform formats buy accuracy. But they change the decoding step. There is no longer a formula that turns a code into a value with one multiply-add. You need a lookup table: the four-bit code is an index, and the value lives in a sixteen-entry table. And lookups have their own physics.

Where do you keep the table? It needs to be fast and shared by the threads that use it, so it goes in shared memory, the small on-chip scratchpad each streaming multiprocessor has. Shared memory is divided into thirty-two banks, each four bytes wide, and in one cycle each bank can serve one address. If several threads in a warp read the same word, the hardware broadcasts it for free. If several threads read different words that happen to live in the same bank, those reads are serialised, one after another. That is a bank conflict.

Now think about a table lookup during decoding. Each thread's index depends on the data: whatever code the weight happened to have. So the addresses are effectively random. A small table of sixteen half-precision values fits in eight words, so conflicts there are limited. But the moment you make the table bigger, or make each entry wider to return more values at once, random indices from thirty-two threads will collide in the same banks, and the lookups serialise. And there is a throughput question even without conflicts: at full memory speed on an H100, a single shared-memory lookup per four-bit weight is already enough to use up roughly all of shared memory's throughput. The lookup is not a side detail. It sits right on the critical path.

One more distinction before we get to FLUTE, because the phrase "lookup table kernel" is used for two opposite ideas. In the first, which is FLUTE's family, the table maps weight codes to weight values: it replaces the decoding formula, and the tensor cores still do the multiplying. In the second, used by systems such as LUT-GEMM on graphics processors and T-MAC on phones and laptops, the table stores pre-computed partial sums of the activations, and the table lookups replace the multiplications themselves. The second idea is elegant, and on processors without tensor cores it can be excellent. On graphics processors, though, it cannot use the tensor cores at all, and later measurements found it running roughly twice as slow as a well-built decode-then-multiply kernel. Remember that; we will come back to it when we talk about hardware.

## FLUTE, piece by piece

Now we have everything we need to understand FLUTE, which stands for a flexible lookup table engine. It is a matrix-multiplication kernel for exactly the case we have been building towards: weights stored in lookup-table formats, including non-uniform ones like normal-float, at four bits and at the awkward three bits, with sixteen-bit activations, for small batch sizes where decoding is memory bound.

FLUTE's authors identify three obstacles, and each of its main ideas addresses one.

The first obstacle is the one we met with the tensor-core layout and the awkward bit width. FLUTE's answer is offline restructuring of the weight matrix. Before the model is ever run, the quantized weights are permuted so that after they are loaded and looked up, the resulting values land directly in the register positions the tensor-core instruction expects. No shuffling at run time.

And for three bits, FLUTE does something more interesting. Rather than packing three-bit codes end to end, where they straddle word boundaries, or padding them to four bits, where they waste bandwidth, it splits each three-bit code into two separate streams: a one-bit stream holding the top bit of every code, and a two-bit stream holding the bottom two bits. Each stream, on its own, has a power-of-two element size. So each stream can be loaded with ordinary, aligned, full-width, coalesced loads, exactly the kind the memory system is built for. Inside the kernel, in registers, a couple of bit operations stitch each code back together from its two pieces. The total bytes moved are exactly three bits per weight, with no padding, and the loads are as efficient as if the format had been two bits or four.

Pause on how different that is from the obvious story. The obvious story asks how many bits per weight. FLUTE keeps the bit count fixed at three and changes the physical representation, the arrangement of bits in memory, so that the representation matches what the hardware can move efficiently. The model is the same. The file is the same size. The bytes are just in a different place.

The second obstacle is the cost of the lookups themselves. FLUTE's answer is vectorised lookup. Instead of a table of sixteen single values, it builds a table indexed by pairs of codes. At four bits, sixteen possible codes means two hundred and fifty-six possible pairs, and each entry holds two half-precision values packed into one thirty-two-bit word. That table is a little over a kilobyte, still small enough for shared memory, and now one lookup returns two decoded weights at once. The number of lookups per weight halves, which matters a great deal when lookups were already sitting on the critical path.

But that creates the third obstacle, which is bank conflicts. A two-hundred-and-fifty-six-entry table spreads across all thirty-two banks, and with data-dependent indices, threads collide. With a single copy of the table, FLUTE's authors describe up to eight-way conflicts at four bits. Their answer is duplication: keep several copies of the table, placed so that different threads tend to read from different banks. Shared memory is a scarce resource, so the number of copies is a tuned trade-off, not a fixed rule. You spend a little on-chip memory to buy back lookup throughput.

There is one more piece, and it is about keeping the whole chip busy. At small batch sizes, and especially after compression shrinks the weight matrix, a matrix multiplication may not have enough independent output tiles to give every streaming multiprocessor its own work. Some of the chip sits idle. FLUTE uses a work decomposition called Stream-K, which splits the long inner dimension of the product across processors, so that several processors each compute part of one output tile and then combine their partial sums. The combining is done carefully: partial sums accumulate in thirty-two-bit precision in registers but are written out in sixteen-bit to save memory traffic. It is one more place where saving bytes shaped a decision.

Finally, FLUTE pairs the kernel with a small refinement to the format. Its "learned normal-float" variant learns one scale factor per tensor on calibration data and folds it into the existing group scales, so the stored format and the kernel are completely unchanged. At four bits on an eight-billion-parameter Llama model, the perplexity it reports is within about a tenth of the unquantized model.

So what does all this buy? At the level of the kernel, FLUTE reports speedups of two to four times over the existing kernels for lookup-table formats, at batch sizes below about thirty-two, sometimes approaching the four-times ceiling. End to end, generating with an eight-billion-parameter model at four bits and batch size one, it roughly doubled throughput on an A6000 workstation card. At three bits, the gain on that card reached about two and a half times.

And then there is a result I want you to notice, because it foreshadows the rest of the lecture. On the A100, a card with much faster memory, the same four-bit model gained only about a third. Same kernel, same format, same model. Faster memory means each byte saved is worth less, and the decode costs, which do not shrink with the memory, become a larger share of the total. Compression and compute trade places depending on the hardware.

FLUTE's authors are candid about the limits. The kernel was tuned for the Ampere generation, and the newer H100 was not yet optimised. Below a batch of sixteen, inputs have to be padded to fit the tensor-core tile shape. Performance falls off at larger batches. It was still behind the best uniform four-bit kernels on the A100. And its configurations were tuned per matrix shape. Most interestingly, the authors end by asking for hardware changes: support for multiplying mixed data types directly, and faster indexing of small tables. We will come back to that request.

## The same move, elsewhere

FLUTE is a clean example, but the move it makes, changing the physical representation to suit the hardware without changing what the model means, shows up again and again.

Consider six-bit weights. A team at Microsoft built a kernel, called FP6-LLM, for six-bit floating-point weights, and ran into exactly the alignment wall we described: six bits do not divide the natural load sizes. Their solution was to split each six-bit weight ahead of time into a two-bit piece and a four-bit piece, stored in separate, aligned regions, in exactly the order the warp would consume them. Then they dequantized four weights at a time within a thirty-two-bit register, using bit-parallel tricks. Same idea as FLUTE's three-bit split, arrived at independently. Their measurements showed the arithmetic units going from being badly under-used to several times better utilised, and the kernel running at about twice the speed of half precision.

Consider two-bit codebook methods, where the gap between a good decoder and a bad one is widest. These methods, which include QuIP-sharp, AQLM and QTIP, quantize groups of weights together as vectors from a codebook, which is how they preserve quality at just two bits per weight. But a codebook is a lookup table, and the size of that table decides everything. QuIP-sharp built its codebook from a lattice with so much symmetry that a codebook of sixty-five thousand entries can be decoded from a table of just two hundred and fifty-six, about a kilobyte, small enough to sit in the fastest on-chip cache. QTIP went further and designed its codes so that each weight can be computed from its bits in two or three ordinary instructions, with an optional two-kilobyte table replicated thirty-two times to avoid bank conflicts. These methods reach a large fraction of peak memory bandwidth. AQLM's most accurate mode, by contrast, uses a codebook of about a megabyte per layer, which does not fit in fast cache, and we will see in a minute what that costs.

Consider the central processor world, where there are no tensor cores at all. T-MAC, which runs low-bit models on laptops and phones, splits weights into one-bit planes and keeps a sixteen-entry table in a single vector register, using the processor's own table-shuffle instruction to do the lookups. It reports kernel speedups of several times over a strong baseline. And, in a detail that makes the whole point of this lecture in one line, without its memory-layout work it was up to about seventeen percent slower than that baseline. The lookup idea alone was not enough. The layout made it work.

And even for the most ordinary format of all, uniform four-bit integers, the kernel dominates the outcome. When the original GPTQ authors rewrote their three-bit kernel, the speedup of the same quantized model on the same hardware went from about one point nine times to about three and a quarter. Same bits, same model. Different kernel, different result.

## When smaller is slower

Now for the claim that sounds most surprising: sometimes a slightly larger representation is actually faster. Here is the evidence.

Start with the starkest case. AQLM at two bits, in its most accurate mode with the megabyte-sized codebook, moves roughly seven or eight times fewer weight bytes than half precision. Yet on a consumer graphics card, with its purpose-built kernel, it generated only about twenty percent faster than half precision. And in one widely used general-purpose implementation, measured by another research group on a different consumer card, it was actually slower than half precision: about twenty tokens per second against about thirty-three. The reason given was simple: the codebook is too big to fit in the fast cache, so every lookup goes further out into the memory hierarchy. When AQLM's authors switched to a mode with two small codebooks that do fit, the speed nearly doubled, at the cost of slightly lower accuracy. Same bit width. The representation, not the bit count, decided the speed.

Next, three bits versus four bits, in the popular llama dot c-p-p block formats. On a seven-billion-parameter model on a consumer graphics card, the three-bit variant, although its file was about a fifth smaller, took about eighteen and a half milliseconds per token, against about fifteen and a half for the four-bit variant. On a laptop processor the gap was larger still. And comparing two three-bit formats with exactly the same number of bits per weight, one based on a codebook needing several table loads per register ran at less than half the speed of the simpler one.

Next, six bits versus four bits. In the FP6-LLM measurements on an A100, the six-bit kernel ran a few percent faster than a well-known production four-bit kernel at batch sizes of eight and sixteen. Fifty percent more bits, slightly less time.

Next, going below four bits for activations as well as weights. The team behind QServe put it bluntly: reducing bit precision does not necessarily speed up inference. They found that a published four-bit weight and four-bit activation system ran twenty to twenty-five percent slower than production systems using four-bit weights with sixteen-bit activations, or eight-bit weights and activations. The reason was the one we already know: in the four-bit-everything scheme, partial sums had to be dequantized inside the main loop, on the slow general-purpose units. By their accounting, one operation on those units costs as much time as fifty four-bit tensor-core operations. They found the same pattern for the attention cache: a naive four-bit cache was about a fifth slower than an eight-bit one.

And there is a fair counterexample, which is just as instructive. For the ExLlama family of kernels, lower bit rates really are faster: three bits per weight beats four, which beats five, on the same consumer card. That is not a contradiction. It is the thesis working in the other direction. Those kernels were built for those formats. When the representation and the kernel are designed together, fewer bits do mean more speed. The trouble only starts when the format is chosen for size and the kernel is left to cope.

## When batching changes the answer

Everything so far assumed a single user and a batch size of one. That is where weight quantization shines, because decoding is so deeply memory bound. But real serving systems batch many requests together, and batching changes the arithmetic.

When a batch of several sequences shares one pass over the weights, each weight fetched from memory is used once per sequence. The arithmetic intensity rises in proportion to the batch size. With sixteen-bit weights on an H100, you would need a batch of roughly three hundred before the matrix multiplications stop being memory bound.

Now here is the irony. Compressing the weights to four bits divides the bytes by four, which multiplies the arithmetic intensity by four. So the batch size at which you hit the compute ceiling drops by four, to around seventy-five. Compression makes you compute bound sooner. And once you are compute bound, the bytes you saved no longer matter, but the decode instructions you added still do. At that point the dequantization overhead is pure cost.

You can watch this happen in the measurements. Marlin gets close to the ideal four-times speedup up to batch sizes of about sixteen to thirty-two, then its advantage shrinks steadily, to about one and a half times at a batch of one hundred and twenty-eight. In a full serving system, on a seven-billion-parameter model, the end-to-end gain went from nearly three times at batch one to about one point two times at a batch of one hundred and twenty-eight. The QServe team's roofline analysis puts the crossover on an A100, where four-bit weights with sixteen-bit activations stop beating eight-bit weights with eight-bit activations, at under about eighty concurrent sequences. And one kernel library states outright that once compute bound, four-bit weight-only kernels are no faster than plain half precision.

Prefill is the extreme case. When a model processes a long prompt, the whole prompt is one enormous batch, and the work is compute bound from the start. In one measurement of a thirteen-billion-parameter model processing prompts on an A100, half precision processed about two hundred and thirty tokens per second, a common four-bit format about a hundred and seventy, and another about a hundred and forty. The compressed models were slower, because in the compute-bound regime you only pay for decoding and gain nothing from the saved bytes.

So the right representation depends on the regime. A format that is excellent for single-user decoding can be the wrong choice for a heavily batched server or for prompt processing. The executable representation is not a property of the model alone. It is a property of the model, the hardware, and the workload together.

## Hardware that speaks the format

All of this has been about software working around the hardware. The last few years show the hardware moving towards the software.

The most important step is block-scaled floating point. An industry group published the microscaling formats: small floating-point elements, in eight, six or four bits, grouped into blocks of thirty-two elements that share a single eight-bit power-of-two scale. The four-bit version costs four and a quarter bits per element once the scale is counted. NVIDIA's Blackwell generation adds its own four-bit variant with blocks of sixteen elements, an eight-bit floating-point scale per block, and a single thirty-two-bit scale per tensor, which comes to about four and a half bits per element.

What makes these formats different from everything we have discussed is not their bit count. It is that the tensor cores consume them directly. On Blackwell, the matrix instruction reads the packed four-bit elements and applies the block scales inside the instruction itself. There is no unpacking on the slow general-purpose units, no two-instruction budget, no magic-number trick. The decode cost goes to essentially zero because the decoder is now part of the tensor core. And because the multiplication itself happens at four bits, the arithmetic rate goes up too: on NVIDIA's own figures for its B200 systems, four-bit tensor throughput is twice eight-bit, which is twice sixteen-bit. The newest parts push four-bit to three times eight-bit, while cutting eight-bit integer throughput sharply, which tells you where the hardware makers think the future is: block-scaled floating point, not integers.

There is a quieter example that makes the thesis almost literally. Blackwell's asynchronous copy engine can keep six-bit and four-bit data packed tightly in main memory, where bytes cost bandwidth, and pad each group out to an aligned hundred-and-twenty-eight-bit slot as it copies into on-chip shared memory, where the tensor cores need regular addresses. That is exactly the split we have been describing: compact where you pay for bytes, aligned where you pay for access, with the conversion happening in the data path. The hardware designers built FLUTE's lesson into the copy engine.

Researchers have also proposed going further. The LUT Tensor Core proposal, evaluated in simulation rather than silicon, designs a tensor core around table lookups over activation partial sums, the second kind of lookup kernel we mentioned earlier. It reports a unit about a sixth the area of a conventional tensor core and large speedups for very low-bit weights. Whether that idea reaches production is open. But notice that it answers FLUTE's closing request for hardware that indexes small tables quickly and multiplies mixed types directly.

The trend is clear. The formats that win in the long run are the ones hardware can consume directly. And the formats hardware chooses to consume are the ones whose decode is cheap: power-of-two element sizes, simple shared scales, small blocks. That is the same set of properties the best software kernels have been fighting to achieve.

## A design rule for our own hardware

So let's turn this into a rule we can use, whether we are choosing a format for an existing processor, writing a kernel, or designing our own hardware.

The wrong question is: what is the lowest bit width we can support? That question optimises the file, and we have seen repeatedly that the file is not what runs.

The better question is: what representation moves the fewest bytes per useful computation, while still being cheap to decode? Let me unpack that into a checklist.

First, count every byte. Bits per weight must include scales, zero points, codebooks, outlier indices, and padding. A nominal three-bit format with heavy metadata may move as many bytes as a clean four-bit one.

Second, check the decode budget. At the memory-bound operating point, how many instructions per weight can the decode spend before it becomes the bottleneck? On current data-centre graphics processors, the answer is a couple of cheap operations. If a format needs table lookups, big codebooks, or conversions, measure whether it fits.

Third, respect the native grain. Element sizes that divide the natural load sizes are cheap. For sizes that don't, bit-slicing into power-of-two streams, as FLUTE and FP6-LLM do, beats both end-to-end packing and padding.

Fourth, lay out for the consumer. Store weights in the order the compute unit wants them, arranged offline for the memory transaction size, the warp, and the matrix instruction's fragment layout. The natural order of the original matrix is irrelevant at run time.

Fifth, keep tables small, and on the fast side of the memory hierarchy. A table that fits in registers or a single bank-friendly slice of shared memory is fast. A table that spills out of cache can erase the benefit of compression. Replicate tables to avoid bank conflicts if you can afford the space.

Sixth, know your regime. Single-user decoding rewards compression almost linearly. Heavy batching and prompt processing are compute bound, and there the decode overhead is pure cost, unless the hardware multiplies in the compressed format directly.

And seventh, if you are designing the hardware, give it the decoder. Native block-scaled formats, mixed-input matrix instructions, fast small-table indexing, and copy engines that repack data on the way on-chip are what turn a clever format into a free one.

Put all of that together and you get the sentence this lecture has been building to. Sometimes a slightly larger representation is faster, because it aligns better with the hardware: six bits that decode in a couple of instructions beat four bits that need a pile of them; four bits with a small table beat two bits with a big one; a plain four-bit format that the tensor core reads directly beats a cleverer one that it cannot. The goal is not the smallest model. The goal is the cheapest executable representation.

## What to remember

Let me close with the key ideas, and then some questions to test yourself.

Decoding a single token is memory bound by more than two orders of magnitude, so bytes moved per token set the speed limit. Quantization lowers that limit, but only if the compressed bytes can be consumed efficiently. Every byte counts, including metadata. The decode step lives on the slow general-purpose units and has a budget of roughly two cheap operations per weight. Memory is read in aligned chunks of fixed sizes, so bit widths that do not fit those chunks need either bit-slicing or padding. Tensor cores need data in a fixed register layout, so weights should be arranged offline. Lookup tables in shared memory are limited by bank conflicts and throughput. FLUTE combines bit-sliced three-bit storage, paired lookups, duplicated tables, and careful work splitting to make lookup formats fast. The same ideas appear in six-bit kernels, in lattice and trellis codebooks, and on laptop processors. Smaller formats can be slower when their decode is expensive, and batching erodes the advantage of weight-only compression. Hardware is converging on block-scaled formats that the tensor cores consume directly.

Now, ten questions, with brief answers after each.

One. Why is single-user decoding memory bound? Because a matrix-vector product does about one operation per byte of half-precision weights, while the processor can do a few hundred operations per byte of bandwidth.

Two. Roughly what speed ceiling does bandwidth impose on an eight-billion-parameter model in half precision on a high-end data-centre card, and what happens at four bits? About two hundred tokens per second at sixteen bits, rising to about eight hundred at four bits, before real-world efficiency losses.

Three. Why is a four-bit model rarely four bits per weight? Because scales, zero points, codebooks and outlier indices add metadata, typically an eighth to a half of a bit per weight or more.

Four. Why is dequantizing in a separate kernel so harmful? Because it writes the expanded weights back to memory and reads them again, which moves more bytes than not quantizing.

Five. What is the decode budget per weight, and why is it so small? A couple of cheap integer operations, because decoding runs on the general-purpose units, which are an order of magnitude slower than the tensor cores.

Six. Why are three-bit weights awkward to load, and how does FLUTE solve it? Three does not divide the natural load sizes, so values straddle words; FLUTE splits each code into a one-bit stream and a two-bit stream, each of which loads cleanly, and reassembles them in registers.

Seven. What problem does FLUTE's paired lookup table solve, and what problem does it create? It halves the number of lookups by returning two values per read; the larger table causes shared-memory bank conflicts, which FLUTE reduces by duplicating the table across banks.

Eight. Why did the same FLUTE kernel give a much smaller speedup on the A100 than on the A6000? Because the A100's faster memory makes each saved byte worth less, so fixed decode costs become a larger share of the total.

Nine. Give an example where a lower-bit format ran slower than a higher-bit one, and explain why. AQLM's most accurate two-bit mode barely beat, or even lost to, half precision because its large codebook did not fit in fast cache; or llama dot c-p-p's three-bit format ran slower than its four-bit format because of its more complex decode.

Ten. What question should replace "what is the lowest bit width we can support?" What representation moves the fewest bytes per useful computation while still being cheap to decode, on this hardware, for this workload.

That is Lecture 1a. Next time, in Lecture 1b, we will look closely at trellis quantization, one of the formats designed from the start around cheap decoding. Thanks for listening.
