# Trellis Quantization: Trading Memory for Cheap Compute

### Lecture 1b in the Quantization and Kernels series

*Approx. 65 minutes spoken. Written to be listened to: numbers are rounded and said in words, the few formulas are described rather than written, and benchmark figures are the ones reported by the paper authors and project maintainers, on the hardware they name. The lecture builds from the hardware up. First why moving bytes is the expensive part of generating a token, then how quantization has tried to move fewer of them, then the trellis idea itself, and finally what happened when two community projects tried to make it fast on real machines.*

---

## Cold open

Here is a strange trade to consider. Suppose you could make a large language model smaller in memory, but only by making the processor do more work every time it reads a weight. Not a little more work, either. Instead of reading a number and using it, the processor would read a short string of bits, run it through a tiny pseudo-random number generator, massage the result with some bit tricks, and only then have the weight it wanted.

On the face of it, that sounds like a bad deal. We are adding arithmetic to every single weight in a model with tens of billions of them.

And yet, on a modern graphics card generating one token at a time, this trade can make the model faster. Not just smaller, faster. The reason is that during that kind of generation, the processor's arithmetic units are mostly sitting idle, waiting for memory. Work that fits inside that idle time is close to free. Bytes you do not have to move are not free at all.

That is the idea behind today's subject: trellis quantization, as introduced by a paper called QTIP from Cornell, and as adapted by two community projects, ExLlamaV3 and ik llama dot cpp. We will see that the idea works beautifully on some machines and poorly on others, and that the difference tells us something general. The lesson is not "use trellis quantization." The lesson is that if your hardware has spare integer or bit manipulation capacity, you may be able to spend it to move far fewer bytes. And if you get to design the hardware yourself, that trade might be worth building in.

We will get there in steps. First the systems side: why generating a token is limited by memory, and how much spare arithmetic is really lying around. Then the software side: how quantization has evolved, why low-bit formats hit a wall, and how a trellis gets past it. Then the practice: what the community found when they ran it on real graphics cards, CPUs, and Apple chips.

## Part one — the memory wall at batch one

Let's start with what a language model actually does when it produces one token for one user. People call this batch-one decoding.

Almost all of the work in a transformer layer is matrix multiplication. When you are processing a single token, each of those multiplications has a very particular shape: a big weight matrix times a single vector. That is called a matrix-vector product. Every weight in the matrix gets read from memory, multiplied by one number from the input vector, added into a running sum, and then never touched again for the rest of that token.

Count what that costs. Each weight gives you two floating point operations, one multiply and one add. If the weight is stored in sixteen-bit floating point, that is two bytes. So you get two operations per two bytes, or one operation per byte moved. That ratio, operations per byte, is called arithmetic intensity. There is a handy rule of thumb for it here: the arithmetic intensity is roughly sixteen divided by the number of bits per weight. Sixteen-bit weights give you one. Four-bit weights give you four. Two-bit weights give you eight.

Now compare that to what the hardware can do. The standard way to reason about this is the roofline model, from Williams, Waterman, and Patterson in two thousand and nine. The idea is simple. A processor has a peak rate of arithmetic and a peak rate of memory bandwidth. The speed you actually get is the smaller of two things: the peak arithmetic, or the bandwidth multiplied by your arithmetic intensity. The point where those two limits meet is called the ridge point, and it is just peak arithmetic divided by bandwidth.

Take a consumer card, the RTX forty ninety. It has roughly one terabyte per second of memory bandwidth, and something like a hundred and sixty-five trillion half-precision tensor operations per second. Divide one by the other and the ridge sits at around a hundred and sixty operations per byte. A data centre H one hundred has more than three times the bandwidth, but also far more arithmetic, so its ridge is up near three hundred.

Batch-one decoding with sixteen-bit weights lives at an arithmetic intensity of one. Even with two-bit weights, it lives at eight. That is a tiny fraction of the ridge on every machine that matters. In roofline terms, you are deep on the sloped, memory-bound side of the roof. The arithmetic units could do fifty or a hundred times more work than you are asking of them, and it would not change the speed at all, because they are waiting on memory.

This gives a clean upper bound on generation speed. Since every weight has to be read once per token, the best possible tokens per second is the memory bandwidth divided by the size of the weights. Let's make that concrete with a seventy-billion-parameter model on that forty ninety. In sixteen-bit form the weights are a hundred and forty gigabytes, which does not fit at all, and even if it did, you would be capped at about seven tokens per second. At four bits it is thirty-five gigabytes, still too big for a twenty-four gigabyte card. At two bits it is seventeen and a half gigabytes, it fits, and the ceiling rises to nearly sixty tokens per second.

So in this regime, the number of bits per weight is the speed. Halve the bits and you roughly double the ceiling. That is why so much effort goes into squeezing weights below four bits, where things get genuinely hard.

## Part two — bytes are the expensive part

There is a second way to see the same wall, and it matters especially for anyone designing hardware: energy.

In twenty fourteen, Mark Horowitz gave a well-known talk at the solid state circuits conference titled "Computing's energy problem." Its numbers, for an older forty-five nanometre process, have been reproduced in countless papers since. A thirty-two-bit integer addition costs about a tenth of a picojoule. A thirty-two-bit integer multiply, about three picojoules. A thirty-two-bit floating point multiply, a little under four. Reading thirty-two bits from a small on-chip memory, about five. And reading thirty-two bits from off-chip DRAM? About six hundred and forty picojoules.

Sit with that ratio. Fetching a word from main memory costs roughly six thousand times as much energy as adding two integers, and a couple of hundred times as much as multiplying them. Song Han's group summarised it as memory access being three orders of magnitude more expensive than simple arithmetic.

Modern memory is better, but the gap persists. Measurements from NVIDIA researchers put graphics memory of the GDDR5 generation at around fourteen picojoules per bit, and stacked high-bandwidth memory at around four picojoules per bit. Either way, a sixteen-bit weight fetched from DRAM costs tens to hundreds of picojoules, and a two-bit weight costs an eighth of that. The savings from not moving those fourteen extra bits could pay for dozens of simple integer operations.

There is a caveat, and it will come back later. On a general-purpose processor, the cost of an instruction is not just its arithmetic. It is also fetching the instruction, decoding it, reading and writing registers, and scheduling it. Horowitz estimated that overhead at tens of picojoules per instruction on a conventional CPU, which dwarfs the add itself. Graphics cards and vector units amortise that overhead across many lanes at once. And a fixed-function hardware decoder removes it almost entirely. So the question "is decoding cheaper than moving bytes?" has a different answer depending on how the decoding is done. Hold on to that.

## Part three — the idle integer units

Let's be more precise about the spare arithmetic, because "the processor is idle" is too vague to design around.

A modern NVIDIA streaming multiprocessor has separate pipelines for different kinds of work. There are floating point pipelines, there are tensor cores for matrix math, and there are integer pipelines. On recent generations, including the Ampere consumer cards, Ada, and Hopper, the integer pipe can issue sixty-four thirty-two-bit integer operations per clock per multiprocessor. And the important detail is that a full thirty-two-bit integer multiply-add runs at that same rate as a simple integer add. Multiplication is not a second-class citizen on these chips.

There are also some unusual instructions that turn out to matter. One is called LOP3. It computes any logical function of three inputs in one instruction, chosen by an eight-bit lookup code. So "mask these bits, then flip these others" can be a single operation. There is a byte permute instruction. There are bit-field extract and insert operations. And there is a family of small integer dot product instructions, the best known being one called dp4a, which takes four bytes from one register and four bytes from another, multiplies them in pairs, and adds all four products into an accumulator, in one go.

Now let's do a budget. Take the forty ninety again. It has a hundred and twenty-eight multiprocessors, each issuing sixty-four integer instructions per clock, at around two and a half gigahertz. That is about twenty trillion integer instructions per second. Meanwhile, at two bits per weight, its terabyte per second of bandwidth delivers about four trillion weights per second. Divide one by the other: you can afford roughly five integer instructions per weight before the integer pipe becomes the bottleneck, and you get a similar number of floating point slots on top of that.

Do the same sum for the H one hundred and something interesting happens. It has more bandwidth, so it delivers far more weights per second, but its integer throughput is not proportionally higher. The budget drops to a little over one integer instruction per weight. Keep that in mind, because the faster the memory, the tighter the decoding budget becomes. A trick that is free on a consumer card can become the bottleneck on a data centre card with faster memory.

There is one more wrinkle on NVIDIA hardware. Converting an integer to a floating point number with the native conversion instruction is slow, a quarter the rate of ordinary integer work. That is why quantization kernels have long used a trick, described by Kim and colleagues at Microsoft in twenty twenty-two and used in the widely deployed Marlin kernel. You take the few bits of a small integer and use a single LOP3 to splice them into the mantissa of a carefully chosen half-precision constant. The result is a valid floating point number equal to a known offset plus your integer. One subtraction removes the offset. No conversion instruction needed. We will see exactly this style of bit trickery at the heart of the trellis decoder.

So the systems picture is this. At batch one, memory is the wall. Arithmetic, and integer arithmetic in particular, is plentiful but finite, and the exact budget depends on the ratio of integer throughput to bandwidth on each chip. Now let's look at how software has tried to exploit that.

## Part four — the scalar grid and where it breaks

The simplest form of weight quantization is scalar. You take each weight on its own and round it to the nearest point on a small grid of allowed values, with a scale factor shared by a group of weights. That is round to nearest. At four bits, with sensible group sizes, it holds up. At two or three bits it falls apart.

Two famous improvements kept the scalar grid but got smarter about rounding. GPTQ, from Frantar and colleagues, quantizes a layer one column at a time. It uses a sample of real activations to estimate how sensitive the layer's output is to each weight, a quantity we call the proxy Hessian. After rounding each column, it pushes the rounding error onto the columns that have not been rounded yet, so later choices compensate for earlier ones. AWQ, from Lin and colleagues, notices that a small fraction of input channels carry large activations and matter far more than the rest. It scales those channels up before quantizing, and folds the inverse scale into the activations, so the important weights get a finer effective grid.

Both are excellent, and at four bits they are hard to beat for the price. But neither can escape a basic geometric fact about rounding one number at a time.

Think of each weight as a point on a line. A scalar quantizer chops the line into intervals and replaces each point with the centre of its interval. Now picture two weights at once, as a point in a plane. Two independent scalar quantizers chop the plane into little squares. In three dimensions, cubes. The question is: how efficient are cubes at covering space? And the answer is: not very. For a fixed number of cells, cubes leave more distance, on average, between a point and its cell centre than rounder shapes would.

Information theory tells us exactly how much we are leaving on the table. If we could use perfectly spherical cells in very high dimensions, the ultimate improvement over cubes is a factor of two pi e over twelve in mean squared error, which is about one and a half decibels, or roughly a quarter of a bit per weight. That is called the granular gain, and a quarter of a bit per weight sounds small until you remember that at two bits per weight, it is a twelve and a half percent budget increase for free. And at low rates there is an additional gain from placing the codepoints where the weights actually are, which is even larger.

To collect that gain, you have to stop quantizing one weight at a time.

## Part five — vector quantization and the codebook explosion

Vector quantization quantizes a group of weights together. You pick a codebook, a list of allowed vectors, and for each group of, say, eight weights, you store the index of the closest codebook vector. At two bits per weight and eight weights per group, that index is sixteen bits long, so the codebook has sixty-five thousand five hundred and thirty-six entries.

That works, and it is how the strongest low-bit methods of twenty twenty-four got their results. But notice the problem. The codebook size is two to the power of the bits per weight times the dimension. Eight weights at two bits is two to the sixteen. Sixteen weights at two bits is two to the thirty-two, about four billion vectors. You cannot store that, and you certainly cannot search it. Brute-force encoding costs as much as the codebook is large. So practical vector quantization has been stuck at around eight dimensions, which collects only part of the available gain.

And even at eight dimensions, the codebook has a systems cost. AQLM, from Egiazarian, Alistarh, and colleagues, uses learned additive codebooks. In its typical two-bit setting, one codebook of sixty-five thousand entries, each eight half-precision values, comes to about a megabyte. That does not fit in the small, fast memory next to the arithmetic units. So during decoding, every group of weights triggers a lookup into a table that lives in slower memory, and those lookups collide with each other. In the QTIP paper's measurements, a two-bit AQLM model of seven billion parameters ran at about eighty tokens per second on a card where the uncompressed model ran at about fifty-six. Faster than full precision, but far below what the reduced size should allow.

This is a systems lesson worth pausing on. The graphics card's shared memory is split into thirty-two banks. If two threads in a group hit the same bank at once, they are serialised. A codebook with hundreds or thousands of entries, randomly indexed, produces exactly those collisions. A study of vector quantization kernels presented at the high performance computer architecture conference in twenty twenty-five found that simply parking whole codebooks in shared memory cost more than thirty percent of compute utilisation through lost occupancy. Lookup tables are not free just because they are on chip.

QuIP sharp, from the same Cornell group that later produced QTIP, found a clever way around the table size. It used the E8 lattice, the densest known packing of spheres in eight dimensions. Because the lattice is so symmetric, you can generate all sixty-five thousand codewords from just two hundred and fifty-six stored vectors plus sign flips and a small shift. The stored part is about one kilobyte and fits in the fastest cache. QuIP sharp reached over a hundred and eighty tokens per second for a two-bit seven-billion-parameter model on that same card. But it was still eight dimensions. To go higher, you need a structure that does not need a table at all.

## Part six — making weights look Gaussian

Before we get to that structure, we need one more ingredient, because it is what makes everything after it possible.

Real weight matrices are messy. Most weights are small, but a few are large, and some channels are much larger than others. A fixed codebook designed for one distribution will fit these poorly. The Cornell group's answer, introduced in the original QuIP paper and refined in QuIP sharp, is called incoherence processing.

Here is the idea. Take the weight matrix and multiply it on both sides by random orthogonal matrices. Orthogonal means they rotate without stretching, so they can be undone exactly. After this random rotation, every entry of the new matrix is a mix of many original entries. Outliers get smeared across the whole matrix. By a central limit argument, the rotated weights look very much like independent draws from a bell curve, a Gaussian. In QuIP sharp, the rotation is a randomised Hadamard transform: a Hadamard matrix, which only contains plus and minus ones, combined with random sign flips. It can be applied in time proportional to n log n, so the rotation itself is cheap, and at inference time it can be applied to the activations instead of the weights.

This matters for two reasons. The first is quality: no outliers means no weight gets a terrible rounding error. The second is the one that matters for us. Once every weight matrix looks like it came from the same Gaussian, you no longer need a codebook tailored to each layer. You can design one quantizer for Gaussian data and use it everywhere. And a quantizer designed for a known, fixed distribution can be generated rather than stored.

Hold that thought. We now have all the pieces on the software side: a need for high-dimensional quantization, a problem with codebook size, and a guarantee that the data is Gaussian. The missing piece comes from an unexpected place: nineteen-eighties modem design.

## Part seven — a trellis, borrowed from modems

In the early nineteen-eighties, Gottfried Ungerboeck showed how to transmit more reliably over noisy telephone lines using what he called trellis-coded modulation. In nineteen ninety, Michael Marcellin and Thomas Fischer turned the idea around. Their paper, "Trellis Coded Quantization of Memoryless and Gauss-Markov Sources," pointed out that transmitting signals and compressing signals are dual problems. The same structure that makes a signal robust to noise can make a quantizer efficient.

Here is what a trellis is, in words. Picture a machine with a fixed number of states. At each step, you are in some state, and you spend a few new bits to choose which way to go next. Each choice leads to a new state and produces one output value. After many steps, the sequence of bits you spent has traced a path through the states, and the sequence of output values along that path is your reconstruction.

Now think about what that gives you. The set of all possible paths over, say, two hundred and fifty-six steps is a codebook of two hundred and fifty-six-dimensional vectors. It has an astronomically large number of codewords, yet you never store them. They are implied by the state machine. The trellis is a way to get the benefits of a huge vector quantizer with the bookkeeping of a small one.

How do you find the best path for a given sequence of weights? That is the job of the Viterbi algorithm, the dynamic programming method that decodes convolutional codes in every phone. You walk through the sequence one step at a time. For each state, you remember only the single best path that ends there, and its total error. At the next step, each state looks at the few states that can lead into it, picks the best, and discards the rest. When you reach the end, you pick the best final state and trace back. The cost grows linearly with the length of the sequence and with the number of states, instead of exponentially with the dimension.

How good is it? For a Gaussian source, the granular gain of trellis-coded quantization grows with the number of states. With a few states it already beats small lattices. With two hundred and fifty-six states it gets within about a sixth of a decibel of the one and a half decibel limit, which is better than even the twenty-four-dimensional Leech lattice, one of the most celebrated objects in mathematics. And encoding stays linear in length.

So why was this not used for language models before twenty twenty-four? Two reasons, both systems reasons. First, a general trellis needs its structure stored: which state goes where on each choice of bits, and which output value each edge produces. For the large number of states you want, those tables are too big for fast inference. Second, and worse, decoding is sequential. To know which state you are in at step one hundred, you have to walk the path from step one. A graphics card wants to decode thousands of weights independently, in parallel, in whatever order its matrix tiles need them. A trellis that must be walked from the start is useless for that.

## Part eight — the bitshift trellis

This is the central idea of QTIP, by Albert Tseng, Qingyao Sun, David Hou, and Christopher De Sa, presented at NeurIPS in twenty twenty-four.

QTIP picks a trellis whose structure is so simple that it does not need to be stored, and where every step can be decoded independently. They call it the bitshift trellis.

Here is the construction. The state is simply a window of sixteen bits, which means sixty-five thousand five hundred and thirty-six possible states. To take one step, you shift the window over by k bits, dropping the oldest k bits off one end and letting k new bits in at the other. At two bits per weight, each step shifts in two new bits. That is the entire transition rule. The next state is the current state shifted, plus the new bits.

Now look at what this means for the compressed data. Lay all the bits for a sequence of weights end to end in one long string. The state at any step is just a sixteen-bit window into that string, starting at a position you can compute directly. To decode the hundredth weight, you do not walk from the first. You jump straight to bit two hundred, read sixteen bits, and you have the state. Every weight depends only on a contiguous sixteen-bit window of the stream. Adjacent weights share fourteen of their sixteen bits, and moving from one to the next is just a shift.

That solves both problems at once. The structure of the trellis costs nothing to store, because it is implied by bit shifting. And decoding is random access and fully parallel: every thread in a graphics card can decode its own weights, from its own window, with no dependence on any other thread.

QTIP applies this to tiles of sixteen by sixteen weights, two hundred and fifty-six weights per trellis sequence, which lines up neatly with the sixteen by sixteen tiles that tensor cores operate on. So the effective dimension of the quantizer is two hundred and fifty-six, compared with eight for the E8 lattice. Thirty-two times larger.

There is one more subtlety worth knowing about. A trellis path has to start somewhere, and naively you would need to store an extra sixteen-bit starting state for every sequence, which is wasteful. QTIP instead uses a tail-biting trellis: the path wraps around, so the final window overlaps the initial one and the bitstream is treated as a circle. That makes every sequence exactly two bits times two hundred and fifty-six weights, with no overhead. Finding the optimal wrap-around path exactly is expensive, so QTIP uses a neat approximation. Rotate the sequence by half its length, run Viterbi, read off the bits in the middle, then run Viterbi again on the original sequence, constrained to start and end with those bits. Two passes, near-optimal result.

The encoding side is expensive but tolerable. Viterbi with sixty-five thousand states over each tile is a lot of work, but it is done once, offline, when the model is quantized, and it parallelises well on a graphics card. It is also why this is a weight-only method. Nobody is running sixty-five-thousand-state Viterbi on activations or on the key-value cache while serving.

## Part nine — a codebook made of instructions

We have a trellis with no stored structure. But each state still needs an output value: the reconstructed weight. With sixty-five thousand states, the obvious approach is a lookup table of sixty-five thousand half-precision numbers, which is a hundred and twenty-eight kilobytes. Too large for the fastest on-chip memory, and full of bank conflicts. We would be back to AQLM's problem.

This is where incoherence processing pays off. Because the rotated weights are Gaussian, the output values do not need to be learned for each layer. They just need to look like a good spread of Gaussian samples. And a spread of pseudo-random Gaussian-looking numbers is something you can compute from the state with a few instructions, instead of looking up.

QTIP calls these computed codes, and it offers three. Let me describe the most elegant one, called three-instruction, or 3INST.

Step one: take the sixteen-bit state and run it through one step of a linear congruential generator. That is one integer multiply-add, with carefully chosen constants, producing a thirty-two-bit number whose bits are well scrambled. On NVIDIA hardware, as we saw, that is a single full-rate instruction.

Step two: treat those thirty-two bits as two sixteen-bit halves, each of which will become a half-precision floating point number. Apply a mask that keeps the sign bit and some of the low bits of each half, and XOR in a magic constant that pins the exponent into a narrow range. Mask and XOR together are exactly the kind of three-input logic that LOP3 does in one instruction. The result is two half-precision numbers, each with a random sign and a magnitude in a bounded range. Their distribution is roughly a two-sided exponential shape.

Step three: add the two halves together. One packed half-precision add. The sum of two such values is very close to a Gaussian.

Three instructions, and you have a pseudo-random Gaussian-looking weight from a sixteen-bit window, with no memory access beyond reading the compressed bits. It is exactly the fast integer-to-float trick from Part three, turned into a random number generator.

The second computed code, called one multiply-add, or 1MAD, runs the same generator, then sums the four bytes of the result. The sum of four roughly uniform bytes is roughly bell-shaped, by the central limit theorem, and it can be centred and scaled into a weight. It costs a few more instructions but is conceptually even simpler.

The third, called HYB for hybrid, is the one QTIP actually shipped in its fastest kernels. It uses a cheap hash of the state to index a tiny table of just five hundred and twelve pairs of values, two kilobytes in total, and one bit of the hash flips a sign. Each lookup produces two weights, so the cost comes to about two instructions per weight. Two kilobytes is small enough to copy thirty-two times, once per bank, so there are no bank conflicts. And unlike the pure computed codes, the table entries can be fine-tuned.

Here is the payoff. On a synthetic Gaussian source at two bits per weight, the best possible mean squared error, the rate-distortion bound, is one sixteenth, or about zero point zero six three. A scalar quantizer, the Lloyd-Max quantizer, gets about zero point one one eight. The E8 lattice codebook of QuIP sharp gets about zero point zero eight nine. The trellis with the three-instruction code gets about zero point zero six nine. The QTIP authors put it this way: the trellis closes the gap between QuIP sharp and an optimal two-bit quantizer by more than a factor of three. And the computed codes do essentially as well as a truly random lookup table would, while storing nothing.

Let me restate what just happened, because it is the entire lecture in miniature. The codebook has been replaced by computation. A table that would have required memory traffic, cache capacity, and bank-conflict-free access has been turned into three instructions on units that were sitting idle anyway.

## Part ten — what QTIP bought

So how does it do on real models?

On quality, the headline results are on the Llama two family, with a context length of four thousand and ninety-six, after the same fine-tuning procedure used by QuIP sharp. For the seventy-billion-parameter model at two bits per weight, the Wikitext perplexity was about three point seven for QTIP, against about three point nine for QuIP sharp and three point eight for AQLM. The full-precision model scores a little over three point one, so QTIP recovered a meaningful share of the remaining gap. For the seven-billion-parameter model at two bits, QTIP scored about five point nine against six point two for QuIP sharp. At three and four bits the improvements were smaller, as you would expect when every method is already close to full precision.

Perhaps more striking: without any fine-tuning at all, the two-bit seven-billion model went from a perplexity of about eight point two with QuIP sharp to about six point eight with the trellis. The better quantizer alone did most of the work.

On speed, the paper measured batch-one decoding on an RTX six thousand Ada, a workstation card with just under a terabyte per second of bandwidth. The full-precision seven-billion model ran at about fifty-six tokens per second. QTIP at two bits ran at about a hundred and eighty-eight, slightly ahead of QuIP sharp, and more than twice as fast as AQLM. At three bits about a hundred and sixty, at four bits about a hundred and forty. The authors' own summary is that QTIP matches QuIP sharp's throughput with a quantizer thirty-two times higher in dimension. The extra decoding work fit inside the idle time.

One honest note on the seventy-billion model. At two bits it ran at about twenty-three and a half tokens per second, where the bandwidth ceiling for seventeen and a half gigabytes on that card is around fifty-five. So even this kernel was reaching a bit over forty percent of the roof. There is still a lot of overhead between "memory bound in principle" and "memory bound in practice," and we will see that gap return in the community implementations.

The group later released a follow-up called YAQA, which improves the rounding step rather than the quantizer. It replaces the layer-by-layer error estimate with an approximation of how each layer affects the whole model's output distribution, and it reduces the divergence from the original model by about thirty percent compared with the older rounding methods. It works with QTIP's trellis. So the Cornell stack is really three separable parts: a rotation, a rounding algorithm, and a quantizer. The trellis is the quantizer.

## Part eleven — ExLlamaV3: turning a paper into a format

A research kernel for one family of models is not the same thing as a format people actually run. That is where community projects come in, and the most direct adaptation of QTIP is ExLlamaV3, from the developer known as turboderp. Its format, called EXL3, is described in the project's own words as a streamlined variant of QTIP.

The core is recognisably QTIP. Weights are split into sixteen by sixteen tiles of two hundred and fifty-six weights. Each tile is a tail-biting trellis with a sixteen-bit state, and the decoder reads a sixteen-bit window that slides by k bits per weight. The bit rate per tensor can be anything from one to eight bits, and a budget allocator mixes rates across tensors to hit fractional targets like two and a quarter, or four, bits per weight overall. There is a Hessian-based rounding step with error feedback, and a fused Viterbi kernel. The practical headline is that the whole conversion runs in one pass, in a couple of minutes for small models and a few hours for seventy-billion-parameter models, on one high-end consumer card.

The deviations are instructive, because each one is a systems decision.

First, the rotation. QTIP rotates whole rows and columns. EXL3 rotates blocks of a hundred and twenty-eight weights along each axis. The maintainer's reasoning is that this helps kernel fusion, and it means a tensor can be split across several graphics cards at a granularity of a hundred and twenty-eight rows or columns without requantizing. A small loss of mathematical purity buys flexibility in how the model is deployed.

Second, and more interesting, the codebook. EXL3 started with QTIP's exact three-instruction generator. Then it added a variant called MCG, which drops the additive constant from the random number generator and keeps only the multiply. Then it added one called mul1, which multiplies, then sums the four bytes of the result using the dp4a instruction, the four-way byte dot product. In mid twenty twenty-six, mul1 became the default for new models.

Why? Here is the clever part. In mul1, the reconstructed weight is an affine function of a byte sum. And dp4a computes a sum of four byte products. So if you quantize the activations to eight-bit integers, you can compute codebook value times activation, for four bytes at once, in a single dp4a instruction. The decoding of the weight and the multiplication by the activation fuse into one integer instruction. The format evolved toward the instruction the hardware does best.

Third, the kernel. The main matrix multiplication kernel needs an Ampere-generation card or newer. It uses asynchronous copies into shared memory and tensor core instructions of a specific shape, and the trellis decoding happens in registers, feeding the tensor cores directly. Older cards lack the right tensor core shapes and synchronisation features.

Now, the part of this story that matters most for our lesson. Early in EXL3's life, users with RTX thirty ninety cards, the consumer Ampere generation, reported that EXL3 generation was slower than simpler formats. One user on three of those cards measured a large model at about seven tokens per second in the older EXL2 format and about three in EXL3. The maintainer's explanation was direct: EXL3 uses an algorithm that is more GPU-intensive and does not easily saturate memory bandwidth on Ampere. Simple integer formats are much easier to unpack on the fly.

In other words, on that card the decoding did not fit inside the idle time. It became the bottleneck. And the fix, which the maintainer later described as much improved, came from finding which specific instruction was the problem. On consumer Ampere, tensor core multiply-accumulate with thirty-two-bit floating point accumulation runs at half rate, and at batch one that was dominating. Accumulating in sixteen-bit instead made batch-one generation about fourteen percent faster on the thirty ninety.

Then consider an even more telling measurement from a user profiling the kernels. On a thirty ninety, the half-precision decoding kernel reached about eighty percent of peak memory bandwidth, which means it was properly memory-bound. On an H two hundred, a data centre Hopper card with vastly more bandwidth, the same kernel reached only about thirty percent. It was compute-bound. The eight-bit integer path, with its fused dp4a decode, made no difference on the thirty ninety, and gave about fourteen percent more end-to-end speed on the H two hundred.

Remember the budget from Part three: about five integer instructions per weight on a consumer card, and barely more than one on a Hopper-class card. This is that budget, showing up in the wild. The same code, on a faster-memory card, runs out of decoding time. The project now even sets a different limit per architecture for when to use the integer path, precisely because the balance point differs between Ampere and Hopper.

On quality, EXL3's published charts are persuasive at low bit rates. On an eight-billion-parameter Llama three point one model, at three bits per weight, EXL3's divergence from the original model was about a fifth of the older EXL2 format's. At four bits, it was about a quarter of EXL2's, and a little over half that of the comparable llama dot cpp i-quant. At higher bit rates, all good formats converge and the differences shrink. The maintainer is candid, too. An older version of the format notes says a faithful QTIP implementation would likely match or beat EXL3 on accuracy, and that the GGUF i-quants hold up well against state-of-the-art formats. EXL3 is a set of engineering compromises in favour of speed and practicality, and it says so.

## Part twelve — ik llama dot cpp: one idea, many machines

The second project takes a very different path, and it runs on far more kinds of hardware. ik llama dot cpp is a fork of llama dot cpp maintained by Iwan Kawrakow, who designed many of the quantization formats in the original project. Its trellis types are called IQ one KT through IQ four KT, at roughly one and three quarters, two and an eighth, three and an eighth, and four bits per weight.

The first thing to know is that these are not QTIP. When he introduced them in late twenty twenty-four, Kawrakow wrote that apart from borrowing the three-instruction generator, the implementation had nothing else in common with QTIP. There is no Hadamard rotation and no tail-biting Viterbi search. Weights are organised in blocks with block scales, like other llama dot cpp formats, and each small group of weights gets a seed, found by a clustering search, that drives the generator for a few steps. You can think of it as a procedurally generated vector codebook: a trellis in name and generator, but not a sliding-window trellis like EXL3's.

His initial verdict on quality is worth hearing: trellis-based quantization is a small improvement over the project's existing formats, but nowhere near the hype. He measured it as needing about a fifth of a bit fewer per weight for the same error. Real, but not revolutionary.

What makes this project so useful to study is that it ran the same idea across CUDA, x86 CPUs, ARM CPUs, and Apple GPUs, and wrote down what happened on each.

On an NVIDIA card, the original floating point generator worked well. On a forty eighty, the two-bit type ran at about a hundred and ninety tokens per second on a seven-billion-parameter model. That was a little slower than the project's simplest two-bit format, but in the same range.

On CPUs, it was a different story, and Kawrakow predicted it. The generator produces floating point values, but fast CPU matrix kernels work on eight-bit integers. Converting every generated float to an integer costs more than the trellis saves. When the CPU port landed, its author noted, as predicted, the CPU ops are very slow. One early port generated tokens barely faster than the uncompressed model.

So he redesigned the generator for integers. The new version multiplies the state by a constant, masks each of the four bytes of the result down to six bits, and sums the four bytes, then subtracts a fixed offset. The result is an integer between about minus a hundred and twenty-six and plus a hundred and twenty-six, roughly bell-shaped. And summing four bytes of a register is precisely what an integer dot product instruction does against a vector of ones. On NVIDIA that is dp4a. On AMD Zen four and Intel with the vector neural network extensions, there is an instruction that does eight of these at once. On ARM, the signed dot product instruction. The same idea, landing on each platform's cheap integer dot product.

Notice the convergence. In a separate thread, a contributor proposed dropping the additive constant and using a particular thirty-two-bit multiplier. Kawrakow adopted it. That same multiplier constant is the one EXL3 locked in for its MCG codebook, and EXL3's mul1 also moved to a masked byte sum computed by dp4a. Two independent projects, starting from QTIP's floating point trick, ended up at nearly the same integer design, because that is what the hardware rewards.

The integer redesign paid off substantially. On a Ryzen seventy-nine fifty X desktop, for an eight-billion-parameter model, token generation went from about eight tokens per second to about fourteen, and prompt processing roughly doubled. On the forty eighty it was slightly faster too, and it had a bonus: it allowed quantized integer matrix multiplication on the GPU, which avoided numerical overflow problems some models had when decoding to half precision.

But on Apple hardware, the story went the other way. The new integer trellis was slower than the original on the Apple GPU. On the M two Max CPU, token generation for the trellis types was around ten to thirteen tokens per second, where other formats of similar size ran noticeably faster. For the one-and-three-quarter-bit type, Kawrakow did not bother to write a Metal implementation at all, because trellis performance on Metal was so low. And as of this month, the four-bit trellis type is simply turned off on Metal, with those tensors falling back to the CPU.

His summary of the whole experience, from a discussion earlier this year, is blunt and worth quoting nearly in full. The trellis quants have good performance on a GPU. On the CPU, it depends. On Zen four or better, performance is reasonable but still lower than other types. On vanilla AVX two, it is noticeably slower. On Apple Silicon, performance is, in his word, pathetic. Elsewhere he gave the advice in one line: don't ask Apple Silicon to do too much work with a piece of data fetched from memory.

Even on NVIDIA there is a price. One user compared four-bit formats on a thirty ninety with a twenty-seven-billion-parameter model. The trellis type had measurably better perplexity than a similar-sized non-trellis type, but generated about twelve percent slower. Their summary: quality per bit, trellis wins; speed, the simpler formats win. And Kawrakow notes that the advantage of trellis quants shrinks as bits per weight rise. At four bits, things are no longer clear-cut.

## Part thirteen — why the same code runs differently everywhere

Let's put the systems side and the community results together, because they explain each other.

A trellis decoder is a short, fixed recipe: a thirty-two-bit integer multiply, some masking, and then either a floating point conversion trick or a byte sum. Whether that recipe is cheap depends entirely on how a given chip executes those particular instructions, compared with how fast it can fetch bytes.

On recent NVIDIA graphics cards, as we saw, a thirty-two-bit integer multiply-add runs at full integer rate, LOP3 does arbitrary masking in one instruction, and dp4a sums bytes in one instruction. The integer pipe is mostly idle during batch-one generation. The recipe fits the budget. Trellis works.

On Apple's GPUs, published microbenchmarks by the developer Philip Turner found that a thirty-two-bit integer multiply runs at a quarter of the rate of an integer add, and that shifts and bit extraction are also slow, sharing a pipeline with transcendental functions. Put that next to the bandwidth. A top Apple chip has around half a terabyte per second. By a rough estimate, at two bits per weight, that leaves room for several simple operations per weight, but less than one thirty-two-bit multiply per weight. The trellis recipe starts with a multiply. So on Apple's GPU, decoding becomes the bottleneck, not memory. And the integer redesign, which leaned harder on integer operations, made it worse there even as it made things better elsewhere.

On x86 CPUs, the vector thirty-two-bit multiply on many Intel cores has a latency of about ten cycles, so a decode chain that depends on it stalls unless there is a lot of independent work in flight. AMD's Zen three and Zen four cores do the same multiply in about three cycles, and have the eight-way byte dot product instruction. That is exactly the split Kawrakow saw: Zen four reasonable, vanilla AVX two noticeably slower. On the other hand, a desktop CPU has only around a hundred gigabytes per second of memory bandwidth, so its per-weight compute budget is comparatively generous. That is why the CPU results improved so much once the decode used integer dot products, and why with enough cores, token generation became nearly memory-bound again.

On ARM CPUs, a vector multiply runs at half rate, while table lookups and logic operations are cheap. That is why lookup-table methods like Microsoft's T-MAC and bitnet dot cpp do so well on ARM. They lean on the fast table lookup instruction rather than multiplies. A trellis generator leans the other way.

And then there are the data centre GPUs, where the problem is not that integer work is slow, but that memory is so fast that the integer budget per weight shrinks to about one instruction. That is why EXL3's decode kernel was compute-bound on the H two hundred, and why the fused integer dot product path helped there and nowhere else.

So here is the general principle, stated carefully. A compressed format has two costs: the bytes you move, and the instructions you run to reconstruct each weight from those bytes. It is fast when the second cost fits inside the time the first cost takes anyway. That depends on three things about the hardware: its bandwidth, the throughput of the specific instructions your decoder uses, and whether those instructions compete with anything else. The same format can be memory-bound on one chip and compute-bound on another. A representation is not fast or slow in the abstract. It is fast or slow on a machine.

## Part fourteen — prefill, the other half of the bill

Everything so far has been about generating tokens one at a time. But every request also has a prefill phase, when the model processes the whole prompt at once. And prefill is a completely different regime.

During prefill, each weight is reused across hundreds or thousands of prompt tokens. The arithmetic intensity is multiplied by the number of tokens, which puts it far past the ridge point. Prefill is compute-bound. The integer and floating point units that were idle during generation are now busy doing the actual matrix multiplication. Every decoding instruction now competes directly with useful work.

You can see this even with simple formats. On Apple's M four Max, in published llama dot cpp benchmarks for a seven-billion-parameter model, going from sixteen-bit weights to four-bit weights made token generation about two and a half times faster, but made prompt processing about four percent slower. Quantization pays off almost entirely in generation, and its decoding cost shows up in prefill.

For heavy formats like the trellis, the answer both community projects reached is amortisation. Don't decode each weight per token. Decode a whole block of weights once into ordinary half-precision or thirty-two-bit floats, then run a standard high-performance matrix multiply over the block with all the prompt tokens. EXL3 reconstructs full tensors and then multiplies whenever the batch is large enough. ik llama dot cpp added a dequantize-then-multiply path for CPUs that more than doubled prompt processing speed for the trellis types in one step, and later added repacking into a simpler eight-bit layout. With that, prompt processing on CPUs became, in Kawrakow's word, excellent. Token generation remained the harder half.

So the full design rule for a compressed weight format has two parts. During generation, the decode must fit inside the memory time. During prefill, the decode must be amortised over enough tokens that it disappears into the matrix multiply.

## Part fifteen — the lesson for custom hardware

Now let's step back to the point of all this.

The lesson is not "use trellis quantization." Trellis quantization is a good quantizer. It gets close to the theoretical limit for Gaussian data at two and three bits, and it does that without a stored codebook. But on the wrong hardware it is slower than simpler formats, and at four bits and above its quality advantage is modest.

The deeper lesson is about the trade itself. Moving a byte from off-chip memory costs hundreds to thousands of times more energy than a simple integer operation, and at batch one it is the only thing that sets the speed. If a machine has integer or bit manipulation capacity that sits idle while it waits for memory, then that capacity can be spent to move fewer bytes. The trellis is one especially clean way of doing that spending: a few instructions buy you the equivalent of a two-hundred-and-fifty-six-dimensional vector quantizer with no table.

For a general-purpose processor, you have to take the instruction set as given and choose a decoder that fits it. That is what both community projects learned. On NVIDIA, multiply-add, LOP3, and dp4a are cheap, so use them. On Apple's GPU, thirty-two-bit multiplies and shifts are expensive, so a multiply-based generator is the wrong choice there, and a small table or pure logic might do better. On ARM CPUs, table lookups are cheap and multiplies are not.

For custom hardware, you get to turn this around. You can build the decoder. Think about what the three-instruction decoder actually is in silicon. It is a sixteen-bit shift register window, one thirty-two-bit multiplier, a mask, and an adder. Or, in the integer version, a multiplier, a mask, and a four-input byte adder. That is a tiny block of logic. Placed right next to the multiply-accumulate units, it could expand two bits per weight into a full-precision weight every cycle, with none of the instruction fetch, decode, and register overhead that makes it costly on a general-purpose core. Horowitz's point from Part two comes back here. On a CPU, an instruction costs tens of picojoules of overhead. In a fixed-function decoder, the multiply costs a few picojoules, and the DRAM bits you avoid moving cost that much or more each.

This is not a new idea in hardware design, and the precedents are encouraging. Stanford's EIE accelerator in twenty sixteen ran directly on compressed, weight-shared networks held in on-chip memory, and much of its roughly hundredfold energy saving came from not going to DRAM at all. NVIDIA's A one hundred added hardware compression of data moving between memory and cache for sparse data. Recent accelerator research, like the lookup table tensor core presented in twenty twenty-five, reports several-fold gains in power, performance, and area over designs that dequantize in software. And for chips that keep weights in on-chip memory, such as wafer-scale or SRAM-based designs, memory capacity rather than bandwidth is the scarce resource. There, every bit saved per weight means fewer chips per model.

The trellis adds something specific to that tradition. Its bitshift structure gives random access, because every weight depends only on a fixed-size window of the compressed stream. That is exactly what hardware wants: no sequential walk, no variable-length decoding like Huffman codes, and a decoder whose cost is fixed and known in advance. The computed codes mean there is no codebook memory to size, fill, or keep coherent. And because incoherence processing makes every layer look Gaussian, one fixed decoder serves every layer of every model.

Before committing, there are honest questions a hardware designer would still need to ask. Encoding is expensive, so this suits weights, not activations or the key-value cache. The Hadamard rotations add their own work at inference time, though that work is small. As the bits per weight rise, the quality advantage over simpler formats shrinks, so the trade is most valuable at two and three bits. And the software ecosystem is still fragmented. The trellis formats have not made it into mainline llama dot cpp or vLLM, partly for licensing and partly because the community formats are incompatible with each other and with the paper. An Apple team's experimental trellis support for their own machine learning framework was closed without being merged.

## Where the field is heading

A few threads are worth keeping an eye on.

On the rounding side, the Cornell group's YAQA shows that a better rounding algorithm, aimed at the whole model's output rather than each layer in isolation, stacks cleanly on top of the trellis. EXL3 has already added a two-sided rounding path that cites it.

On the quantizer side, the trellis is not the last word. A team at Qualcomm AI Research published a Leech lattice vector quantizer in March of this year, twenty-four dimensions, reporting better two-bit perplexity than QTIP on the seven-billion Llama two model. Interestingly, its reported error on a synthetic Gaussian source is actually higher than the trellis's, so the gain probably comes from other parts of its pipeline. That is worth watching rather than taking at face value. Other work extends trellis quantization to fractional bit rates and mixed schemes, and one recent preprint makes the trellis differentiable, so a model can be trained while aware of its trellis quantization, by replacing Viterbi's hard choice with a soft average over paths.

On the kernel side, the direction is clear from both community projects: fuse the decode into integer dot products with quantized activations, so that decoding and multiplying become the same instruction. That is the software version of putting the decoder next to the multiplier.

## Recap

Let's pull it together in a few breaths.

At batch one, a model's speed is set by memory bandwidth, because every weight is read once per token and used for just two operations. Fewer bits per weight means proportionally more speed, and moving bytes also dominates energy.

Meanwhile the integer units sit mostly idle. On a consumer NVIDIA card, there is room for about five integer instructions per two-bit weight. On a data centre card with much faster memory, about one.

Scalar quantization wastes about a quarter of a bit per weight because cubes pack space badly. Vector quantization recovers some of that, but its codebook grows exponentially with dimension and lookup tables are costly on real hardware.

Incoherence processing rotates weights so they look Gaussian, which means one fixed quantizer can serve every layer.

A trellis gives a very high-dimensional quantizer with a small state machine and linear-time Viterbi encoding. QTIP's bitshift trellis makes each state a sixteen-bit window into the bitstream, so decoding is random access and parallel. Its computed codes turn the state into a Gaussian-looking weight in about three instructions, with no table.

QTIP got close to the theoretical limit at two bits and ran as fast as the best eight-dimensional method, on a card where the decode fit the idle time.

ExLlamaV3 turned it into a practical format, then evolved its codebook toward the dp4a instruction. It found that the decode is compute-bound on some cards and memory-bound on others.

ik llama dot cpp redesigned the generator for integers and ran it everywhere. It is good on NVIDIA, reasonable on Zen four, slower on older x86, and poor on Apple Silicon, because the cost of the same few instructions varies so much.

Prefill is compute-bound, so heavy formats decode a block once and reuse it.

And the lesson: if your hardware has spare integer or bit manipulation capacity, spend it to move fewer bytes, and choose a decoder that fits your hardware's cheap instructions. If you are building the hardware, a trellis decoder is small, fixed-cost, and random-access, which makes it a strong candidate to build in.

## Self-check questions

Ten questions, with brief answers after each.

One. Why is batch-one decoding memory-bound on essentially every modern processor? Because each weight is used for only two operations per token, giving an arithmetic intensity of roughly sixteen divided by the bits per weight, far below the ridge point of any modern chip.

Two. What is the upper bound on tokens per second at batch one? Memory bandwidth divided by the size of the weights, ignoring the key-value cache and other overheads.

Three. Why can a faster memory system make decoding compute-bound? Because the number of weights arriving per second rises faster than integer throughput, so the instruction budget per weight shrinks. On a Hopper-class card it is about one instruction per weight at two bits.

Four. What limits vector quantization to around eight dimensions? The codebook grows as two to the power of bits per weight times dimension, and both storing it and searching it become impractical.

Five. What does incoherence processing buy, beyond removing outliers? It makes every layer's weights look Gaussian, so a single fixed quantizer, and therefore a computed code, can be used everywhere.

Six. What is the state in QTIP's bitshift trellis, and why does it make decoding parallel? The state is a sixteen-bit window into the compressed bitstream, which shifts by k bits per weight. Any weight can be decoded from its own window without walking the path from the start.

Seven. What does the three-instruction code do? A multiply-add scrambles the state, a single three-input logic instruction masks and XORs it into two bounded half-precision values with random signs, and one add sums them into a roughly Gaussian weight.

Eight. Why did ik llama dot cpp redesign its trellis for integers, and why did that hurt on Apple's GPU? Converting generated floats to integers for fast CPU dot products cost more than the trellis saved. The integer version maps onto byte dot product instructions on CUDA, x86, and ARM. But on Apple's GPU, thirty-two-bit multiplies and shifts are slow, so the extra integer work made decoding the bottleneck.

Nine. How do heavy formats avoid paying the decode cost during prefill? By decoding a block of weights once to ordinary floating point, or repacking it, and reusing it across all the prompt tokens in a standard matrix multiply.

Ten. What is the general lesson for designing custom hardware? Off-chip bytes are far more expensive than simple logic, so a small fixed-function decoder next to the multiply-accumulate units can trade cheap computation for much less memory traffic. The trellis is attractive for that because it is random-access, table-free, and fixed-cost.

That is the lecture. The trellis is a beautiful piece of information theory, but the reason it matters is a systems reason: it is a way to spend arithmetic you already have to avoid moving bytes you cannot afford. Thanks for listening.
