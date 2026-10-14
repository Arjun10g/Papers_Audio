# From Checkpoint to Silicon: How Language Models Run on Real Hardware

Imagine we have just finished training a large language model with eight billion parameters. We have spent an enormous amount of computational power training it on GPUs, adjusting billions of numerical values until it becomes capable of understanding prompts, generating text, writing code, and answering questions.

Now training is complete.

The model exists as a collection of files stored on disk. But we have a completely new problem: how do we transform those files into a running system that can answer thousands of user requests?

And more importantly, how does the computation eventually reach actual silicon? How do technologies such as llama dot C P P, LLVM, RISC-V, CUDA, and GPU kernels work together to make a language model generate its next token?

By the end of this lecture, you should understand the entire process, from a saved model checkpoint all the way down to the instructions being executed by a physical processor.

Let's begin with what training actually produces.

## What Training Actually Produces

When you train a transformer language model, the result is primarily a massive collection of numerical parameters called weights. For example, an eight-billion-parameter model contains approximately eight billion learned numbers.

These numbers are organized into tensors, which are multidimensional arrays. Some tensors represent the weights of attention projections. Others represent the feed-forward networks, token embeddings, normalization parameters, and output projections.

But there is an important distinction between the model's weights and the model's architecture.

The architecture specifies which mathematical operations need to happen and how those operations connect to each other.

The weights supply the learned numerical values used by those operations.

Imagine the architecture as a factory blueprint, and the weights as the precise settings of the machinery inside that factory. Neither one tells a physical CPU or GPU exactly how to perform the work. We still need a working implementation of the computations.

That implementation is the responsibility of an inference runtime.

An inference runtime is software that loads the model, organizes its numerical operations, manages memory, and executes those operations on available hardware.

Examples include llama dot C P P, v L L M, and other inference systems.

For our lecture, we'll concentrate on llama dot C P P because it provides a particularly useful way to study the connection between model mathematics, compiled software, and physical hardware.

Before we get there, we need to understand the model files themselves.

A trained model might originally be stored in a format such as SafeTensors, accompanied by configuration files describing its architecture.

The configuration might specify that the model contains thirty-two transformer layers, that its internal hidden representation has four thousand and ninety-six dimensions, and that its attention mechanism uses thirty-two query heads.

It might also describe the number of key-value heads, vocabulary size, normalization settings, and positional encoding configuration.

These details matter because the inference runtime needs to know the dimensions of every operation.

For example, suppose one layer contains a weight matrix with four thousand and ninety-six input dimensions and four thousand and ninety-six output dimensions.

The matrix contains roughly sixteen point eight million numerical values.

Its associated operation involves calculating many dot products, where each output is produced by multiplying corresponding input values and weights and then adding those products together.

A processor doesn't know that these numbers come from an artificial intelligence model. It only knows how to load numbers, multiply them, add them, and store results.

This is one of the most important ideas in LLM systems engineering.

An LLM is a sophisticated mathematical structure, but at the hardware level, its execution becomes numerical instructions, memory accesses, and communication between computing resources.

Now let's discuss how those numerical weights are stored.

Suppose our eight-billion-parameter model uses sixteen-bit floating-point numbers, meaning each weight occupies two bytes.

Eight billion multiplied by two bytes gives sixteen billion bytes, or approximately sixteen decimal gigabytes.

That is just the storage required for the weights.

It does not include the model's intermediate activations, key-value cache, execution buffers, operating-system overhead, or memory consumed by other processes.

If we want to run this model on a GPU with twelve gigabytes of memory, the original sixteen-gigabyte weight representation will not fit entirely into that GPU.

We therefore have several choices.

We could use a GPU with more memory. We could split the model between devices. We could keep some operations on the CPU. Or we could reduce the size of the model's weights using quantization.

Quantization is the process of representing numerical values using fewer bits.

Instead of storing each weight using sixteen bits, we might store many weights using four bits.

Four bits gives sixteen possible bit patterns. To represent a wider range of real numerical values, a quantization scheme associates those patterns with scales and, sometimes, offsets.

For example, we might divide a weight tensor into small blocks. For each block, we calculate scaling information and represent individual weights using small integers.

During inference, the kernel can reconstruct approximate numerical values or perform specialized arithmetic directly on the packed representation.

The theoretical raw storage for eight billion four-bit weights is four billion bytes, approximately four gigabytes.

Real quantized formats require additional metadata, and some tensors may remain at higher precision, so the final model file will usually be larger than this theoretical minimum.

Importantly, quantization is not automatically free. Lower precision can change the model's outputs, and some hardware may not accelerate a particular quantization format efficiently.

So the goal is not simply to minimize file size. The goal is to achieve a useful balance of memory consumption, output quality, and inference speed.

This brings us to GGUF.

GGUF is a model file format used by llama dot C P P. It stores model tensors along with structured metadata describing the architecture, tensor dimensions, numerical types, and other information required for inference.

A compatible checkpoint can be converted into GGUF and then quantized using supported tools.

Notice what is happening here.

We are changing how the model's parameters are stored, not converting the entire neural network into one enormous machine-code executable.

The GGUF file contains model data. The inference runtime contains the executable code that knows how to use that data.

And converting a checkpoint into GGUF does not guarantee compatibility. The runtime must still implement the model architecture and the required numerical operations.

## Tokens, and One Pass Through the Model

There is one more essential ingredient: the tokenizer.

An LLM doesn't directly operate on written English.

Suppose a user types, "Explain how a GPU works."

The tokenizer transforms that text into a sequence of integer token identifiers.

Those identifiers might represent entire words, fragments of words, punctuation, or other text units.

The model uses its embedding table to convert token identifiers into numerical vectors.

For a chat model, the service also applies a chat template. This adds the appropriate structure identifying the system instructions, user messages, and assistant responses.

Using the wrong tokenizer or the wrong chat template can cause incorrect behavior even if every model weight is loaded correctly.

Now we finally have everything necessary to begin inference.

We have the architecture, weights, numerical representation, tokenizer, and an inference runtime capable of executing the model.

Let's follow a request.

Imagine the user asks, "Why is the sky blue?"

After tokenization, the model receives a sequence of token identifiers.

Each identifier is mapped to an embedding vector.

The model then processes these vectors through a sequence of transformer layers.

Inside a representative Llama-style transformer layer, the input first goes through normalization.

Normalization adjusts the numerical scale of the representation so subsequent calculations operate on appropriately conditioned values.

Next, the model creates queries, keys, and values.

It does this by applying learned linear projections.

A query is a vector used to determine which information the current position should attend to. A key provides information against which queries are compared. A value contains information that can contribute to the resulting representation.

The model compares queries against keys using dot products.

It then scales the results and applies a softmax operation, turning attention scores into a collection of nonnegative weights that sum to one.

Those weights determine how the value vectors are combined.

In a causal language model, positions cannot attend to future tokens.

This makes sense because when generating text, the model must predict what comes next without having access to future output tokens.

Llama-style models also use rotary positional embeddings, which modify the query and key representations to incorporate positional relationships.

Attention is followed by an output projection, a residual connection, and then a feed-forward network that applies additional learned transformations.

The block's output is passed into the next layer.

This repeats until the model reaches its final layer.

The result goes through a final transformation into logits.

Logits are numerical scores associated with possible next tokens.

The generation algorithm might select the token with the highest score, or it might sample from a probability distribution based on those scores.

Suppose the first generated token corresponds to the beginning of the word "Because."

We have successfully generated one token.

But producing a complete answer might require another hundred tokens.

How do we generate those efficiently?

## Prefill, Decode, and the Key-Value Cache

This introduces two extremely important concepts: prefill and decode.

During prefill, the model processes the input prompt.

Suppose the prompt contains two thousand tokens.

All two thousand are already known, which creates opportunities to process many positions in parallel.

The model must still respect causal attention dependencies, but the computations for known positions can be organized into large matrix operations.

This can utilize GPU arithmetic hardware efficiently because many input vectors are processed together.

At the end of prefill, the runtime has the information needed to select the first generated token.

Then comes decoding.

During decoding, the system processes the newly generated token to calculate scores for the next token.

Unlike prefill, it does not know the rest of the response.

It must generate one token, observe the result, and then calculate the next.

For a single sequence using ordinary autoregressive decoding, this creates a sequential dependency between generation steps.

Even if we have thousands of GPU cores, they cannot simply generate all future tokens independently using the usual decoding algorithm.

But there is another important optimization.

Imagine that the model has already processed two thousand tokens.

When generating the next token, should it recalculate every previous attention key and value from scratch?

That would waste substantial computation.

Instead, the inference runtime stores earlier attention keys and values in what we call the key-value cache, or KV cache.

At the next decoding step, the model creates the new token's query, key, and value representations.

The attention computation can reuse earlier cached keys and values, calculate the new output, and extend the cache.

This avoids recomputing those earlier projections.

However, the cache consumes memory, and the current query still needs to interact with the relevant earlier keys and values.

So the KV cache reduces computation, but it introduces another major memory-management problem.

A model's weights are largely static during inference. The KV cache grows as sequences become longer and depends on how many requests are being served.

Keep that distinction in mind, because later it will explain why serving many users requires substantially more memory than merely loading a model once.

## Inside the Inference Runtime

We now understand the mathematical task: take token embeddings, repeatedly execute transformer operations, calculate logits, select a token, and continue.

The next question is how this mathematical description becomes a real executable system.

This is where llama dot C P P enters the picture.

llama dot C P P is a C and C plus plus inference implementation built on top of a numerical computation library called GGML.

GGML provides tensor representations, numerical operations, computation graphs, and hardware backend interfaces.

llama dot C P P provides the logic for executing supported model architectures and generating tokens.

Its server component provides the HTTP interfaces through which applications can submit requests and receive generated responses.

It is useful to think of this system as several layers of responsibility.

At the highest level, the server understands requests and responses.

Below that, the model implementation understands transformer architecture.

Below that, GGML understands tensor operations.

And below that, hardware-specific kernels implement the actual arithmetic.

Let's take one operation as an example.

We have a vector containing four thousand and ninety-six values. We want to multiply it by a matrix containing approximately sixteen point eight million weights.

In the model implementation, we don't necessarily write sixteen million individual multiplication instructions.

Instead, we describe the operation as matrix multiplication.

GGML represents that operation as a node in a computation graph.

The graph is effectively a description of what needs to be calculated and which calculations depend on others.

For example, normalization produces an output tensor.

A matrix multiplication consumes that tensor.

An attention operation consumes the result of that matrix multiplication.

Each node represents an operation, and the connections describe data dependencies.

This is a crucial distinction: constructing a graph is not the same thing as executing it.

The graph describes the work.

The hardware backend determines how the work will actually be performed.

Suppose llama dot C P P is running on an ordinary CPU.

GGML's CPU backend will execute supported operations using CPU numerical kernels.

A kernel is a routine designed to perform a particular numerical computation efficiently.

A matrix-multiplication kernel performs matrix multiplication. A normalization kernel implements normalization. A specialized quantized dot-product kernel can process compressed weights.

In a different deployment, GGML might use a CUDA backend and execute appropriate operations on an NVIDIA GPU.

The model's mathematical architecture does not fundamentally change when we change devices.

What changes is the machinery used to perform its operations.

GGML also has backend scheduling machinery that can allocate graph buffers, assign operations to supported devices, and arrange transfers between backends when required. This is different from the higher-level server scheduler that manages competing user requests.

Now imagine our model uses four-bit weights.

The matrix-multiplication kernel cannot necessarily treat each packed four-bit value like an ordinary thirty-two-bit floating-point number.

Instead, the kernel needs to understand the quantized representation.

It may load packed bytes, extract the individual small integer values, apply block scaling, and accumulate results using a wider numerical type.

The exact implementation depends on the quantization scheme and processor capabilities.

This is one of the reasons llama dot C P P is interesting: its performance depends not just on the transformer architecture but also on a large collection of carefully optimized numerical routines.

And those numerical routines eventually become machine instructions.

## From Source Code to Machine Code

Let's follow that process on a CPU.

Imagine that we are running llama dot C P P on a RISC-V processor.

We have the source code for a dot-product kernel, written in C or C plus plus.

That source code is understandable to developers, but the processor itself cannot execute C plus plus text.

The processor executes machine instructions encoded in a format defined by its instruction-set architecture.

We therefore need a compiler.

This is where LLVM enters our story.

LLVM is compiler infrastructure. It provides representations, optimization techniques, and target-specific code generation capabilities used by compilers.

Clang is a compiler front end that can translate C and C plus plus source code into LLVM intermediate representation.

Intermediate representation, often called IR, is a structured, lower-level form of the program that is useful for optimization.

A simplified compilation process looks like this.

First, Clang parses the source code and checks that it follows the language's rules.

Then it lowers the program into intermediate representation.

LLVM optimization passes analyze and transform this representation.

They may eliminate unnecessary calculations, move invariant computations, simplify expressions, or transform suitable loops into vector operations.

After optimization, the target-specific code generator maps the program into machine instructions.

The compiler must know which processor architecture it is targeting and which optional instruction extensions it can use.

For our example, that target might be sixty-four-bit RISC-V with selected floating-point and vector capabilities.

The compiler selects appropriate instructions, assigns values to registers, inserts loads and stores where necessary, and eventually emits machine code.

The generated object files are then linked with the rest of the program and required libraries to produce an executable.

On a conventional Linux system, that executable may use the ELF binary format.

When the program starts, the operating system's loader maps the executable and required libraries into the process's address space.

The CPU can then fetch and execute the machine instructions they contain.

Notice an important subtlety.

LLVM is involved in compiling the inference software, but it does not have to process the model's eight billion learned weights as source code.

The model's weights remain data that the compiled program loads and uses.

Also, LLVM is not necessarily invoked every time a token is generated. In a conventional ahead-of-time build of llama dot C P P, the relevant CPU kernels were already compiled before the server started.

Some AI frameworks do use just-in-time compilation or runtime graph compilation, but those are additional implementation choices, not requirements for all LLM inference.

## RISC-V and Its Vector Instructions

Let's go deeper into the instructions themselves.

RISC-V is an instruction-set architecture, or ISA.

An ISA defines the software-visible interface of a processor: its instructions, registers, and execution behavior.

The ISA is not the physical processor itself.

Two manufacturers could build very different CPUs implementing compatible RISC-V instruction sets.

One might build a relatively simple core designed for energy efficiency.

Another might design a high-performance processor with sophisticated branch prediction, several arithmetic execution units, large caches, and out-of-order execution.

Both could execute compatible RISC-V binaries while delivering very different performance.

This distinction between architecture and implementation is essential.

RISC-V tells software what operations the processor is capable of performing.

The processor's microarchitecture determines how those operations are carried out internally.

For example, imagine a simple numerical calculation: multiply two values and add the result to an accumulator.

At a conceptual level, the CPU loads the operands into registers, performs the arithmetic, and updates the accumulator.

A register is a small piece of storage directly accessible to processor instructions.

Registers are extremely important because arithmetic units can operate on their contents without repeatedly accessing main system memory.

But there are only so many registers available.

When the compiler needs to keep more live values than fit in the available registers, it may need to temporarily store some values in memory.

This is called register spilling.

Spilling introduces additional memory operations, potentially slowing execution.

That is one reason register allocation is a significant stage in LLVM's code-generation pipeline.

Now suppose our matrix multiplication needs to perform millions of similar operations.

If the CPU executes one multiplication at a time, it may not use its available arithmetic hardware efficiently.

Modern processors therefore provide techniques for parallel execution.

One technique is instruction-level parallelism.

A sophisticated processor can identify independent instructions and potentially execute multiple operations in overlapping time.

Another is thread-level parallelism.

We can divide the computation across several CPU cores, giving different threads different pieces of work.

A third is data-level parallelism, commonly implemented through vector instructions.

And this brings us to one of the most interesting parts of RISC-V for LLM inference: the RISC-V Vector extension, also called R V V.

The vector extension allows one instruction to operate on multiple numerical elements.

Suppose our computation requires multiplying one thousand and twenty-four pairs of numbers.

With a scalar approach, we conceptually handle one pair at a time.

With a vector approach, we can process groups of elements together.

RISC-V vector instructions operate on vector registers.

An important property of R V V is that software can be written to adapt to different vector register widths.

For example, imagine one processor has vector registers that are two hundred and fifty-six bits wide.

If we use thirty-two-bit elements, a single register can hold eight such values.

Another processor might have five-hundred-and-twelve-bit registers, allowing sixteen thirty-two-bit values in the same basic configuration.

The vector extension uses configuration information that includes the selected element width, register grouping, and active vector length.

The active length controls how many elements are processed by an operation.

A program can request work on the remaining elements, receive an active vector length supported by the processor, process that chunk, and repeat.

That means one well-designed loop can operate across different RISC-V implementations without hardcoding a single number of elements per iteration.

LLVM supports this through scalable vector representations and target-specific instruction generation.

Let's connect this to our quantized model.

Imagine a dot-product kernel receives a block of four-bit weights and a block of activation values.

The routine may first load packed weights into vector registers.

Then it uses bit operations to extract their numerical representations.

Next, it combines the extracted values with the activations through vector arithmetic.

It accumulates the results, potentially using wider integer or floating-point representations to avoid excessive rounding or overflow.

Finally, it applies the relevant quantization scales.

This is how compressed weights can be turned into useful numerical computations without first expanding an entire eight-billion-parameter model into a high-precision copy.

However, the compiler cannot invent efficient vectorization in every situation.

The way the data is laid out, the numerical precision rules, dependencies between operations, and the available instructions all influence the generated code.

For performance-critical kernels, developers may use architecture-specific intrinsics or handwritten assembly.

Intrinsics are functions or constructs recognized by the compiler that explicitly request particular low-level operations.

They give the programmer more control over hardware instructions while keeping much of the implementation in C or C plus plus.

It is also important not to assume that wider vector registers always produce proportionally higher performance.

A vector register capable of holding sixteen elements does not necessarily imply the processor can complete sixteen arithmetic operations every cycle.

The physical arithmetic units, execution pipelines, and memory supply all matter.

## The Memory Wall

Let's move from registers to the memory system.

A CPU can execute numerical instructions extremely quickly, but LLM inference frequently requires reading enormous amounts of weight data.

Imagine our quantized model occupies approximately five gigabytes in memory.

The CPU has small high-speed registers, several levels of cache, and a much larger DRAM memory system.

A typical cache hierarchy includes level-one, level-two, and sometimes a shared level-three cache.

These caches keep recently accessed data close to the processor.

But five gigabytes of model weights will usually be much larger than the available CPU caches.

During token generation, the runtime may need to stream substantial portions of those weights from main memory.

And now we encounter a major performance limitation: memory bandwidth.

Memory bandwidth describes how much data can be transferred per second.

Memory latency describes how long an individual memory access takes.

These are different properties.

A processor might have enough arithmetic capability to process a huge number of numerical operations per second, but its arithmetic units can be forced to wait if the necessary operands arrive too slowly.

Let's calculate an idealized example.

Suppose our model requires reading approximately four gigabytes of weights from main memory for one decoding step.

Suppose the system can sustain one hundred gigabytes per second of memory bandwidth for that workload.

Four gigabytes divided by one hundred gigabytes per second equals zero point zero four seconds.

That is forty milliseconds.

Even if all arithmetic were instantaneous, moving those four gigabytes through that memory interface would take at least forty milliseconds, assuming the data really must cross the interface.

One divided by zero point zero four gives twenty-five.

So under those assumptions, the bandwidth-limited upper bound is twenty-five generated tokens per second for this single sequence.

Real execution includes additional arithmetic, synchronization, cache traffic, intermediate tensors, and other overhead.

Also, actual traffic might be different from the model's nominal file size, depending on caching and implementation.

But the example demonstrates the fundamental problem.

The GPU or CPU might be capable of performing more arithmetic, yet the service cannot generate tokens faster because the necessary data cannot arrive quickly enough.

This is also why adding CPU cores does not necessarily double inference speed.

Suppose four cores already saturate the system's memory bandwidth.

Adding four more cores creates additional demand for the same memory interface.

Unless those cores improve data reuse or otherwise reduce the bottleneck, the throughput may barely change.

Sometimes extra threads can even reduce performance through contention or synchronization overhead.

On multisocket CPU servers, the problem becomes more complicated because memory may be physically attached to different processor sockets.

This is called non-uniform memory access, or NUMA.

A thread running on one socket may access memory attached to another socket at a different latency and effective bandwidth.

Therefore, thread placement and memory allocation can materially affect CPU inference.

All of these principles apply whether the CPU uses RISC-V, x eighty-six, or Arm.

The instruction set changes the available operations and compiler targeting, but it does not eliminate the physics of data movement.

We can summarize CPU inference using two fundamental quantities.

The first is arithmetic throughput: how many useful mathematical operations the CPU can perform per second.

The second is memory bandwidth: how much numerical data the CPU can move per second.

We can combine them using the concept of arithmetic intensity.

Arithmetic intensity means the number of arithmetic operations performed per byte of data transferred from a particular level of memory.

If we perform very little arithmetic for each byte loaded from DRAM, the workload is likely to become memory-bandwidth limited.

If we perform substantial arithmetic on reused data, the workload may become compute limited.

This is the central intuition behind the roofline performance model.

It also explains why prefill and decoding behave differently.

During prefill, we process many tokens together.

A block of weights can contribute to calculations for many different token positions.

That provides opportunities for reuse and large matrix-matrix multiplication.

During single-sequence decoding, we are often applying the same large weight matrices to one new token vector.

That produces something closer to matrix-vector multiplication.

The arithmetic intensity can be much lower because there is less reuse of each weight load.

This means the same model, on the same processor, can operate under two very different performance constraints depending on whether it is processing a prompt or generating new tokens.

## How a GPU Works

Now we are ready to understand GPUs.

A GPU is not simply a CPU containing more cores.

It is a processor designed around a different balance of execution resources and parallelism.

A modern CPU devotes substantial hardware to executing individual instruction streams quickly, including sophisticated caching, branch prediction, and out-of-order execution.

A GPU devotes a much larger share of its resources to high-throughput execution across many parallel threads.

The GPU is especially effective when we can divide the work into large numbers of similar numerical operations.

That is exactly what large matrix multiplication provides.

Let's use NVIDIA's CUDA programming model as our example.

A CUDA application contains host code, generally executed by the CPU, and device code, executed by the GPU.

The CPU manages the overall program and launches GPU work.

The GPU receives commands that cause its hardware to execute numerical kernels.

On NVIDIA hardware, GPU execution resources are organized into streaming multiprocessors, or S Ms.

A CUDA kernel is launched using a grid of thread blocks.

Each thread block contains threads that are assigned work.

Threads are organized into groups of thirty-two called warps.

Threads within a warp execute using NVIDIA's single-instruction, multiple-threads execution model, often abbreviated SIMT.

This is related to CPU vector processing, but it is not identical.

A CPU vector instruction typically expresses an operation on multiple elements in a vector register.

CUDA threads have their own logical thread identities and can follow different program paths, although divergent control flow within a warp can reduce efficiency.

Both approaches exploit data parallelism, but they expose it to programmers differently.

Now suppose our transformer needs to multiply two large matrices.

We could assign different groups of output elements to different GPU thread blocks.

Each block handles some portion of the matrix computation.

But the critical optimization is not just parallelism.

It is data reuse.

Suppose we load a tile, or small rectangular block, of a matrix from GPU memory.

Instead of reading the same values repeatedly for every multiplication, we want to reuse those values for several calculations.

A well-designed GPU kernel can load tiles into faster on-chip memory, perform many arithmetic operations, and write the results back.

GPUs provide registers and shared memory for this kind of work.

Shared memory is a fast, programmer-managed memory space that allows threads within a thread block to exchange and reuse data.

Modern GPUs also provide specialized matrix arithmetic hardware, often called Tensor Cores on NVIDIA architectures.

These units can perform matrix multiply-and-accumulate operations very efficiently for supported numerical formats.

This is why a GPU can achieve enormous arithmetic throughput during suitable matrix operations.

However, merely placing a calculation on the GPU does not guarantee high utilization.

A GPU has many parallel execution resources.

If we give it only a small amount of work, much of the hardware may remain underutilized.

This is especially relevant during single-request decoding.

Even though the model performs billions of numerical operations, the execution pattern may not use the available matrix hardware as efficiently as a large batch of token representations.

Another important consideration is GPU memory.

Dedicated accelerator GPUs commonly use high-bandwidth memory, or HBM, while other GPUs may use GDDR memory.

This device memory holds model weights, intermediate tensors, and the KV cache.

It can offer substantially higher bandwidth than conventional server DRAM, depending on the hardware.

But GPU memory is usually a more constrained resource than the host's system memory.

And if model data must move between CPU memory and dedicated GPU memory, transfers can introduce additional costs.

For example, an NVIDIA GPU attached through PCI Express communicates with its host over that interconnect.

PCI Express is much slower than the GPU's internal memory interface.

Consequently, a system that repeatedly transfers large model weights between CPU and GPU during generation may lose much of the advantage of GPU acceleration.

This is why keeping frequently used weights resident in GPU memory is so valuable.

And it explains why quantization can be particularly useful for GPU inference: smaller weights allow more of the model to remain on the accelerator.

## CPUs and GPUs Working Together

Now let's reconnect the GPU to LLVM and RISC-V.

Suppose our host CPU uses RISC-V.

We compile the CPU portion of llama dot C P P to execute RISC-V machine instructions.

But the GPU does not automatically understand RISC-V instructions.

An NVIDIA GPU executes instructions belonging to its own device architecture.

The CUDA toolchain compiles device-specific kernel code through its supported compilation pipeline, involving representations such as PTX and ultimately executable device code.

The CPU and GPU therefore execute different instruction streams.

The RISC-V processor runs the host program.

The GPU runs suitable accelerator kernels.

The host program coordinates the execution.

This is heterogeneous computing: different kinds of processors cooperate within one application.

LLVM can participate in compilation for multiple targets, but there is not one universal instruction stream that every CPU and GPU executes identically.

The actual compilation path depends on the software stack and selected backend.

llama dot C P P supports various hardware backends, including CPU, CUDA, AMD's HIP, Apple's Metal, and others, subject to device and operation support.

Now consider a hybrid setup.

Our model is too large for the GPU, so we leave some layers on the CPU and place others on the GPU.

The CPU-side layers perform their calculations with CPU kernels.

The GPU-side layers execute their calculations with GPU kernels.

At the boundary, the intermediate activations may need to move between host and device memory.

Because later layers depend on earlier layer outputs, those transfers can become part of the critical execution path.

This arrangement allows a model to run when it would otherwise exceed available GPU memory, but it does not automatically give us the performance of a fully GPU-resident model.

The optimal configuration depends on memory capacity, transfer bandwidth, kernel efficiency, and the amount of work assigned to each device.

We have now connected almost the entire system.

The saved model contains numerical weights.

llama dot C P P understands the architecture.

GGML represents operations and dispatches them to hardware backends.

Clang and LLVM can compile CPU kernels into RISC-V instructions.

GPU toolchains compile accelerator kernels into device code.

The CPU coordinates the overall program, and the GPU performs suitable parallel numerical computations.

But there is still a huge difference between running one prompt locally and operating a production inference service.

To understand that difference, we need to examine memory requirements, attention optimization, request scheduling, concurrency, and multi-GPU scaling.

## Memory, Long Prompts, and FlashAttention

Let's start by examining the total memory required to run our model.

We already know that an eight-billion-parameter model stored using four-bit weights occupies roughly four gigabytes before quantization metadata and other overhead.

But the weights are only one part of the memory footprint.

The runtime also needs buffers for intermediate computations, memory for input and output data, and storage for attention's key-value cache.

Depending on the execution configuration, there may be additional temporary buffers for converting numerical formats or transferring data between devices.

The KV cache is particularly important because its size increases with the amount of active context.

Let's calculate its memory consumption for a representative Llama-style model.

Assume our model has thirty-two transformer layers.

It uses thirty-two query attention heads but only eight key-value heads. This arrangement is called grouped-query attention, or GQA.

Rather than storing separate keys and values for every query head, groups of query heads share a smaller set of key-value heads.

This reduces KV cache memory and attention memory traffic compared with using thirty-two independent key-value heads.

Suppose each key-value head contains one hundred and twenty-eight numbers.

Assume each cached number occupies two bytes.

For one token, one layer must store eight key heads and eight value heads.

Each head contains one hundred and twenty-eight numbers.

So we multiply two, representing keys and values, by eight heads, by one hundred and twenty-eight numbers, by two bytes.

That gives four thousand and ninety-six bytes per token, per layer.

Across thirty-two layers, that becomes one hundred and thirty-one thousand and seventy-two bytes for one token.

For a sequence containing two thousand and forty-eight tokens, the total is two hundred and sixty-eight million, four hundred and thirty-five thousand, four hundred and fifty-six bytes.

That is approximately two hundred and fifty-six mebibytes, a little over a quarter of a gigabyte, of KV cache memory.

And remember, that is for one sequence.

If sixteen users each maintain a two-thousand-and-forty-eight-token context, their combined KV cache would require approximately four gibibytes, about four point three billion bytes, under those assumptions.

That is in addition to model weights and other working memory.

We have just discovered an important limitation of production inference.

Even when the model weights fit comfortably on a GPU, serving many concurrent users can exhaust device memory because each active request requires additional state.

Real systems may reduce this memory through lower-precision cache storage, prefix sharing, or other optimizations, but the basic tradeoff remains.

The more active context we retain, the more memory the service generally requires.

Long prompts create another problem.

Recall that attention compares queries against keys.

For a prompt containing two thousand tokens, the attention mechanism conceptually has to consider many relationships between those positions.

A straightforward implementation could create a large attention-score matrix.

For a sequence containing four thousand tokens, a square attention matrix contains sixteen million elements per attention head.

Writing and reading these intermediate scores can require substantial device memory traffic.

And this motivates an extremely important optimization called FlashAttention.

FlashAttention reorganizes the exact attention calculation to reduce movement between the GPU's relatively large off-chip memory and its smaller, faster on-chip memory.

Instead of storing an entire attention-score matrix in device memory, the algorithm processes blocks of queries, keys, and values.

It uses tiled calculations and an incremental softmax method to accumulate the correct attention result without materializing all intermediate scores.

The crucial insight is that the same mathematical operation can be organized in a way that moves substantially less data.

FlashAttention does not magically eliminate the quadratic arithmetic work of ordinary full attention during prefill.

It primarily reduces memory traffic and temporary storage while organizing the computation efficiently for hardware.

This illustrates a general principle of inference optimization.

Sometimes the fastest algorithm is not the one with the fewest arithmetic instructions.

Sometimes it is the one that minimizes expensive data movement.

That applies to CPU caches, GPU shared memory, quantized kernels, attention implementations, and communication between processors.

## Serving Many Users at Once

Now let's move from a single user to a hundred users.

Imagine we have deployed our model behind an HTTP server.

A user opens an application and sends a chat message.

The application sends an HTTP request to the inference service.

The request contains the conversation, the selected model, and possibly generation settings such as temperature and maximum output length.

The server validates and parses the request, prepares the model input, and assigns the work to its inference engine.

The engine processes the prompt and begins generating tokens.

As tokens become available, the server may stream them back through the network.

So far, this resembles our earlier single-user example.

But suppose another user submits a request while the first user is still generating a response.

Should the second user wait until the first request finishes?

That is one possible scheduling policy, but it wastes opportunities for parallel computation.

Instead, a serving engine can group work from multiple requests into batches.

Imagine four users are each waiting for their next token.

Rather than performing four completely separate model evaluations, the engine may evaluate their current token positions together.

The model's weight matrices can then be applied across several token representations in one batched operation.

This increases the amount of useful arithmetic performed for each weight load.

Remember our memory-bandwidth calculation.

Single-sequence decoding may require moving substantial model-weight data to generate one token.

With batching, one weight block can contribute to the calculations for multiple sequences.

The amount of arithmetic increases, but the weight traffic may grow much more slowly than the number of sequences.

That improves arithmetic intensity and can increase aggregate throughput.

There is a tradeoff, however.

A larger batch consumes more KV cache memory and requires more computation in each decoding iteration.

Individual users may experience higher latency even when the system generates more total tokens per second.

This creates two competing objectives.

We want high system throughput, meaning many tokens generated across all requests.

But we also want low user latency, meaning each individual user gets a quick response.

These objectives are related but not identical.

Now imagine user one requests a ten-token answer, user two requests fifty tokens, and user three requests five hundred tokens.

A traditional static batch might keep these requests grouped until the batch finishes.

Once the first user completes, that user's batch position may become wasted.

Continuous batching addresses this inefficiency.

As requests finish, the scheduler can remove them from active execution and introduce new requests.

The active batch changes dynamically over time.

In llama dot C P P, the server supports multi-user inference and continuous batching. Other engines, including v L L M, also implement sophisticated scheduling around the same broad principle.

But scheduling becomes more complex when requests contain very different prompt lengths.

Suppose one user has already finished prefill and is receiving output tokens.

Another user submits a fifty-thousand-token document for summarization.

If the engine dedicates a large uninterrupted block of execution to processing the new document, the first user's next-token latency might increase substantially.

One technique for managing this tradeoff is chunked prefill.

Instead of processing the entire large prompt as one scheduling unit, the engine can divide it into smaller portions.

Those portions can be interleaved with decoding work.

This allows the scheduler to balance long-prompt processing against the need to keep already-active users receiving tokens.

v L L M, for example, documents a scheduling approach that prioritizes decoding and uses remaining token budget for prefill chunks. Exact policies depend on the serving engine and configuration.

## Paged Caches and Speculative Decoding

There is another subtle memory-management challenge.

Each sequence's KV cache grows while the sequence generates tokens.

Different sequences finish at different times.

Some have short contexts, while others have very long contexts.

If memory is allocated as large contiguous regions for each request, the system can experience fragmentation or reserve substantial memory that is not actually used.

A major serving technique for addressing this is paged KV cache management.

The idea is similar to virtual memory in operating systems.

Rather than requiring one large contiguous physical region for a sequence's cache, the serving engine can divide cached token data into blocks.

A logical sequence then references the blocks containing its keys and values.

Those physical blocks do not necessarily need to be adjacent.

This makes allocation and reclamation more flexible and can support sharing cached prefixes between requests.

The PagedAttention work associated with v L L M developed this approach to improve KV cache utilization and serving throughput.

It is important to distinguish this from FlashAttention.

FlashAttention primarily concerns efficiently calculating attention by reducing memory traffic.

PagedAttention concerns efficient attention execution over a paged key-value memory layout.

They address related but different problems.

Let's add another inference optimization: speculative decoding.

Recall that ordinary autoregressive generation creates a dependency between successive output tokens.

A large model predicts one token, then another, then another.

Speculative decoding tries to reduce the effective cost of these sequential steps.

One approach uses a smaller, faster draft model to propose several candidate tokens.

The larger target model then evaluates those candidates together and determines which can be accepted.

With an appropriate verification and sampling algorithm, speculation can preserve the target model's intended output distribution.

If enough proposed tokens are accepted, the system can produce multiple output tokens with less sequential work by the expensive target model.

But speculative decoding is not always beneficial.

If the draft model frequently proposes unacceptable tokens, or its own computation is expensive, the additional work may outweigh the savings.

Again, performance depends on the relationship between hardware utilization, computation, and data movement.

## Scaling Across Several GPUs

Now suppose we have optimized quantization, kernels, batching, and cache management, but the model is simply too large for one GPU.

What happens next?

We have several forms of parallelism available.

The first is data parallelism, or more precisely for serving, model replication.

Imagine we have four GPUs, each capable of holding a complete copy of our model.

We can load one replica onto each GPU.

A load balancer distributes independent requests among the replicas.

This allows us to serve more users, assuming the workload scales appropriately.

But it does not make one individual request four times faster simply because four replicas exist.

Each request still runs on one model replica.

Now imagine a much larger model that cannot fit on a single GPU.

We might instead split one model across several devices.

One method is tensor parallelism.

Tensor parallelism partitions computations within individual model layers.

For example, consider a large matrix multiplication.

One GPU might calculate one portion of the output, while another calculates a different portion.

Depending on how the matrix is partitioned and which operation follows, the GPUs must exchange partial results or gather the required tensor pieces.

That communication introduces overhead.

Fast interconnects such as N V Link can be valuable because tensor parallelism may require communication between devices at many layers.

A second method is pipeline parallelism.

Instead of splitting individual layer computations, we divide the model into stages.

For example, the first group of transformer layers might execute on one GPU, and later layers on another.

An activation passes through the first stage, then travels to the next stage.

For a single request, this may not reduce latency dramatically because the stages still depend on one another.

But with multiple requests or microbatches moving through the pipeline, different stages can work on different pieces of work at the same time.

This is analogous to an assembly line.

One station works on one item while another station works on a different item.

Pipeline execution can improve throughput, although stage imbalance and communication costs can reduce efficiency.

A third technique, particularly relevant to mixture-of-experts models, is expert parallelism.

In a mixture-of-experts architecture, parts of the model contain multiple expert networks.

A routing mechanism selects which experts process a given token.

We can distribute experts across devices.

But this means tokens or intermediate representations may need to travel to the devices holding their selected experts.

Communication can become a major bottleneck, especially if routing produces uneven workloads.

Modern distributed inference systems may combine tensor, pipeline, data, and expert parallelism.

The right combination depends on the model architecture, available device memory, interconnect topology, and workload.

v L L M's scaling documentation, for example, distinguishes single-GPU serving from multi-GPU tensor parallelism and multi-node combinations of tensor and pipeline parallelism.

And this brings us to an important observation.

Adding GPUs does not automatically improve performance.

If the model already fits on one GPU, splitting it across two devices might introduce communication overhead that makes inference slower.

If we have a very large model that cannot fit on one device, splitting it may be necessary.

If our primary problem is serving many independent users, adding replicas may be more effective than splitting individual model computations.

We must understand the bottleneck before choosing the architecture.

## A Real Deployment

Let's now connect everything to a practical cloud deployment.

Imagine we want to host our quantized eight-billion-parameter model on AWS.

First, we select a suitable compute instance.

This is ultimately a hardware decision.

We need to know which CPU architecture the instance provides, how much system RAM it has, whether it includes GPUs, how much GPU memory is available, and what the CPU-to-GPU communication path looks like.

Suppose we choose a Linux machine with a supported NVIDIA GPU.

We install or deploy a llama dot C P P build containing the appropriate GPU backend and required runtime libraries.

We place the GGUF model on storage accessible to the machine.

We then launch the inference server with the model path and suitable execution settings.

At startup, the runtime opens the model file and reads the necessary metadata.

It identifies the architecture, tensor names, dimensions, quantization types, and configuration.

Depending on the setup, model data may be memory-mapped into the process or loaded through other mechanisms.

If GPU execution is enabled, the runtime arranges for supported model tensors and computations to reside on the selected devices.

It also initializes execution buffers and KV cache resources.

The service may perform warm-up computations so that some first-use initialization costs occur before normal traffic arrives.

Then the server begins accepting requests.

For an externally accessible production service, we would normally add authentication, encrypted network connections, logging, monitoring, and appropriate routing or load balancing.

A load balancer may send incoming traffic to different inference workers.

Each worker can maintain a loaded model replica, avoiding repeated weight loading for every request.

If the service is distributed across multiple machines, orchestration software can manage worker processes and their lifecycle.

But notice that none of this changes the core transformer calculation.

The infrastructure manages access to computation.

The inference engine still performs the model's numerical operations.

We could perform a similar deployment on a supported RISC-V Linux machine.

The major difference would be that the host executable must target the RISC-V instruction set, and the available hardware backends must support the machine's actual devices.

We might compile llama dot C P P using Clang and LLVM configured for the appropriate RISC-V extensions.

Or we could use a compatible prebuilt binary if one exists.

The processor still needs adequate memory and computational resources to make inference practical.

Simply selecting RISC-V does not make the model faster, cheaper, or smaller.

RISC-V changes the software-hardware instruction interface. The performance consequences depend on the processor implementing that interface and the quality of the numerical kernels.

## Measuring and Diagnosing Performance

Now let's ask how a production engineer measures whether the serving system is performing well.

One of the first metrics is time to first token.

This measures how long a user waits between submitting a request and receiving the first generated token.

It includes delays such as queueing and prompt processing, along with the work required to produce the first token.

Another metric is inter-token latency.

This describes the interval between generated tokens after generation begins.

If the server produces one token every fifty milliseconds, its output appears relatively smooth.

But if it produces several tokens quickly and then pauses for a long interval, the experience can feel inconsistent even if its overall average throughput is similar.

We also measure output tokens per second.

For one user, this describes how quickly their response is generated.

For the service, aggregate output throughput describes how much generation is completed across all active requests.

These measurements should not be confused.

A system serving a hundred users might produce one thousand tokens per second in aggregate while each individual user receives only a fraction of that rate.

We also care about tail latency, such as the ninety-fifth or ninety-ninth percentile latency, because real users do not all experience average performance.

If a small fraction of requests experience enormous delays, the average can hide a serious serving problem.

In addition to these user-visible metrics, we monitor hardware.

On CPUs, we examine core utilization, instructions executed, memory bandwidth, cache behavior, thread scheduling, and NUMA placement.

On GPUs, we examine device utilization, memory allocation, kernel execution time, occupancy, and memory throughput.

We may also examine data-transfer costs and whether individual kernels are launching enough work to use the GPU effectively.

Let's imagine our model generates only ten tokens per second and we expected forty.

What should we do?

We should not immediately buy a more expensive GPU.

Instead, we need to identify the constraint.

Perhaps the model is partially executing on the CPU, with unexpectedly expensive transfers between devices.

Perhaps its quantized representation is not supported efficiently by the selected GPU kernels.

Perhaps the batch size is too small to utilize the accelerator.

Perhaps the model weights are much larger than anticipated.

Perhaps the GPU is reaching its memory-bandwidth limit.

Or perhaps the inference engine is experiencing excessive synchronization and launch overhead.

Each problem requires a different solution.

If the bottleneck is memory bandwidth, increasing arithmetic capability may provide little improvement.

If the bottleneck is compute throughput, using a faster matrix kernel or a more suitable numerical representation may help.

If the bottleneck is KV cache capacity, reducing the cache's memory requirements or improving cache allocation may increase concurrency.

If the bottleneck is inter-device communication, changing how the model is partitioned may be more valuable than adding additional arithmetic resources.

This is what makes LLM inference a systems-engineering discipline rather than simply an exercise in executing neural-network mathematics.

We need to understand the complete path.

## The Whole Journey

Let's finish by walking through that path one final time, as if we were watching a single request under a microscope.

Our model has finished training.

Its learned parameters have been saved to files, along with information describing the architecture and tokenizer.

We convert the checkpoint into a format supported by our selected runtime and optionally quantize the weights to reduce memory consumption.

Separately, we compile the inference software.

If our target is a RISC-V CPU, a toolchain such as Clang and LLVM can convert the relevant C and C plus plus code into machine instructions for that processor.

If we want GPU acceleration, the selected GPU backend supplies appropriate accelerator kernels compiled for the supported device.

We start the inference server.

The runtime reads the model metadata, prepares tensors, establishes memory allocations, and loads the numerical weights into accessible memory.

Now a user sends a request.

The server parses the conversation and applies the chat template.

The tokenizer converts the prompt into token identifiers.

The model's embedding operation converts those identifiers into numerical vectors.

The runtime organizes the transformer calculations into operations and dependencies.

Its hardware backend dispatches the required kernels.

The CPU performs its assigned work using registers, execution units, caches, and main memory.

On RISC-V, suitable kernels may use vector instructions to process groups of numerical values.

When execution is assigned to a GPU, the CPU launches appropriate GPU work.

The GPU distributes computations across its execution resources, using device memory, caches, registers, and shared memory.

Large matrix operations benefit from parallel execution and data reuse.

The prefill phase processes the known prompt and establishes its initial KV cache state.

The model produces logits for the next token.

The generation procedure selects one token.

The server can send the corresponding text to the user.

Then decoding begins.

Each new token passes through the transformer layers.

Earlier keys and values are reused from the cache.

Model weights are accessed as required.

New attention state is added to the cache.

Another token is selected.

The process repeats.

Meanwhile, other users may be generating tokens.

The serving scheduler groups compatible work into batches, manages available cache memory, and balances ongoing decoding with new prompt processing.

If the server becomes overloaded, requests may queue.

If the model cannot fit on one GPU, its operations or layers may be distributed across several devices.

If more independent request capacity is required, we can add model replicas behind a load balancer.

Throughout this process, the fundamental hardware constraints remain the same.

We have finite arithmetic throughput.

We have finite memory capacity.

We have finite memory bandwidth.

We have finite communication bandwidth between devices.

And we have dependencies that limit how much work can happen simultaneously.

The art of efficient LLM serving is organizing the computation to make the best possible use of those finite resources.

Now let's connect the biggest technologies in one final mental picture.

The transformer architecture defines what mathematics must happen.

The trained weights determine the numerical behavior of that mathematics.

GGUF stores the model data and metadata in a form llama dot C P P can consume.

llama dot C P P implements the inference process, including model execution and token generation.

GGML provides tensor operations, computation graphs, and backend execution infrastructure.

LLVM can compile CPU-side numerical kernels and supporting code into instructions for a target architecture.

RISC-V defines the instruction interface that a compatible CPU implements.

The CPU executes instructions and coordinates the overall application, while GPU backends allow suitable numerical work to run on accelerators.

Memory bandwidth, arithmetic throughput, quantization, batching, and cache management determine much of the system's practical performance.

The serving infrastructure turns that computation into a network-accessible service capable of handling real users.

If there is one concept I want you to retain from this entire lecture, it is that an LLM is not inherently a GPU program, a CPU program, a RISC-V program, or an LLVM program.

It is a learned mathematical model.

An inference system provides an implementation of that model's calculations.

A compiler turns appropriate parts of that implementation into instructions supported by particular processors.

The runtime connects the model's data to those instructions and manages their execution.

And the hardware performs the underlying arithmetic while moving enormous amounts of information through its memory hierarchy.

Once you understand how computation moves from mathematical operations to numerical kernels, from kernels to machine instructions, and from machine instructions to physical execution units, you understand the foundation of modern LLM inference.

That is the complete journey: from a trained checkpoint sitting on disk to a production system generating tokens for users in real time.
