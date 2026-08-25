# The Model Is No Longer Just the Model

### Model–serving co-design in August 2026

*Approx. 25 minutes spoken. A companion to "Same Model, Twice as Fast": that lecture takes a fixed model and squeezes it; this one is about how the model and the machine that serves it are increasingly designed together.*

August 2026 gives us a useful window into where large language model architecture is heading, because the most important developments are no longer about making the neural network smarter. Increasingly, the architecture of the model and the architecture of the serving system are being designed together.

That is the idea to keep in mind throughout.

If you want to understand modern inference, do not begin by asking how many parameters the model has. Ask what has to move through the system every time it produces a token. Ask where the weights live. Ask what has to be remembered about the conversation. Ask how many GPUs have to talk to each other. And ask whether the infrastructure serving the model actually matches the shape of the model.

Those questions explain a remarkable amount of what happened in August.


## The Sequential Problem

Start with the fundamental problem of inference.

When we train a language model, we already know the whole sequence the model is supposed to process, so we can parallelize enormously. During generation we cannot. The model produces one token, uses it to help produce the next, and repeats.

That sequential loop gives inference a completely different performance profile from training.

When serving one user or a few users, the hardware is usually limited not by how much arithmetic it can do, but by how fast it can move weights and state to the places where the arithmetic happens. A modern GPU can execute a staggering number of operations, but if every new token requires dragging huge quantities of weights out of memory, the arithmetic units spend much of their time waiting.

So the theme of the month is reducing how much information has to move per useful generated token.


## Qwen3.8 and Mixture of Experts

The Qwen3.8 family is the clearest example of the co-design philosophy.

The largest model has roughly two point four trillion parameters in total, which sounds absurd. But it does not use all of them for every token. Only about ninety-five billion are active at a time.

The mechanism is mixture of experts. Instead of one enormous general-purpose block, the model contains hundreds of specialist sub-networks called experts. When a token arrives, a small routing network decides which few experts should handle it. The model can hold an enormous amount of total capacity without executing all of it for every token.

That sounds like a pure win, and it creates a new problem, because the experts have to live somewhere. Spread hundreds of experts across many GPUs and tokens now have to be routed across the machine or the cluster. One token wants expert fourteen, another wants expert two hundred and seven, another wants a different combination entirely. The system has to ship tokens to the right experts, run them, and gather the results back.

So mixture of experts reduces computation and increases communication.

This is the principle that governs everything that follows: when you remove one bottleneck, another becomes visible. Reduce arithmetic and communication starts to dominate. Reduce weight traffic and the cache starts to dominate. Reduce the cache and synchronization starts to dominate. No optimization exists independently of the full system.


## The KV Cache

To see why, we need the key-value cache.

When a Transformer generates text, it needs information about the tokens that came before. Recomputing the entire conversation for every new token would be disastrous, so the model stores intermediate information about previous tokens — the keys and values — and attends to those instead. This is what makes autoregressive generation practical.

The catch is that the cache grows with the conversation. A context of a hundred thousand tokens means retaining information for a hundred thousand tokens across many attention layers. Multiply by thousands of simultaneous users, and memory capacity becomes the primary limit on how many people you can serve.

This is why recurrent architectures are interesting again. A recurrent layer does not keep a separate entry for every token in history; it maintains a fixed-size internal state that gets updated as new information arrives.

The analogy is the difference between carrying the complete transcript of a meeting and carrying an evolving summary. The transcript is expensive to store, but you can retrieve precise details from it. The summary is small and cheap, but exact details get compressed away. Full attention is the transcript. Recurrent state is the summary.

Qwen3.8 uses both. Most layers use recurrent state, which is memory-efficient. Periodically, the architecture uses full attention, which restores high-fidelity access across the whole sequence.

That matters because it suggests the future is not a contest between Transformers and recurrent models. It is memory hierarchies inside the neural network itself — some layers offering cheap compressed memory, others offering expensive precise access. The neural-network equivalent of registers, cache, main memory and storage.


## Typed Model State

Here is where serving gets genuinely harder.

A server built for a conventional Transformer can think about request state as essentially one thing: KV cache. A hybrid model like Qwen3.8 has several kinds. There is full-attention KV state. There is recurrent state. There is short sliding-window or convolutional state. There may be speculative-decoding state.

These cannot be treated identically. Some state is reusable whenever two prompts share a prefix. Some recurrent state is only valid at a particular checkpoint. Some can be shared safely; some must be copied before a conversation branches.

So serving frameworks are developing what you might call typed model state. SGLang's Unified Radix Cache is one example. The underlying idea — prefix sharing — is already familiar: if a thousand requests begin with the same ten-thousand-token system prompt, it would be absurd to process those ten thousand tokens a thousand times, so the server stores the state for the shared prefix and reuses it.

The new difficulty is that a hybrid architecture requires the cache to understand semantics, not just matching. The server cannot say "these tokens match, therefore everything is reusable." It has to say: this attention state is reusable, this recurrent state is reusable only to this checkpoint, this sliding-window component needs this trailing region, and this mutable component needs a private copy before we touch it.

That is a much more sophisticated abstraction, and it is one of the biggest conceptual developments of the month. The server is no longer executing a neural network. It is becoming a state-management system for a continuously evolving process.


## Prefill and Decode as Separate Services

There are really two computational phases behind every request.

Prefill processes the prompt you supplied. Twenty thousand prompt tokens can be processed with enormous parallelism; the system is reading the input and building the state it needs before generation starts. Decode is the generation phase: produce a token, update state, produce the next one.

These are very different workloads. Prefill handles many tokens at once. Decode handles a tiny number of new tokens while repeatedly touching a very large model and an ever-growing context.

Historically, serving systems ran both on the same hardware layout. Qwen3.8 and SGLang show why that is becoming unattractive, because the best parallel configuration differs between them.

For prefill you may want pipeline parallelism — divide the model by layers, with one GPU running the early layers, another the next group, and so on. Large batches of prompt tokens flow through that pipeline efficiently.

For decode, mixture of experts makes expert parallelism more attractive, spreading the experts across GPUs so no single GPU has to hold or repeatedly load the entire expert bank.

So the best physical representation of the model can genuinely differ between the two phases. Which leads to a striking idea: treat them as separate services. The prefill service reads the prompt and constructs state. That state transfers to a decode service with a completely different layout tuned for token generation.

Instead of one model server executing one model, we start thinking about a pipeline of specialized services that exchange model state — one for ingesting context, one for generating tokens, perhaps others for speculative drafting, retrieval, or long-term memory.


## Speculative Decoding

Return to the fundamental problem: generation is sequential, and normally the large model runs once per token.

Speculative decoding asks whether we can avoid that. A smaller, cheaper model — the drafter — quickly proposes several future tokens, say six. The large model evaluates all six together. If it agrees with the first four, those four are accepted; at the first disagreement, the rest is discarded and generation resumes from the corrected point.

Four output tokens from roughly one large-model verification, instead of four separate large-model steps. That can be a dramatic speedup.

But it depends entirely on acceptance rate. If the drafter proposes eight tokens and only one survives, most of that work was wasted.

August research adds an important observation: the optimal number of speculative tokens is not fixed. Under light load there is spare compute, verifying extra possibilities is cheap, and speculating aggressively pays. Under heavy load those extra positions compete for scarce resources, and the server wants shorter drafts.

Adaptive systems like DSpark act on this. Rather than fixing a speculative length, the scheduler weighs the probability that each proposed token survives verification against the current cost of verifying it — a token near the start of the draft is likely to be accepted, one far out much less so — and prioritizes accordingly.

That is a real shift. Speculative decoding stops being purely a model algorithm and becomes a scheduling algorithm. The correct policy depends on current server load, which means generation behavior is now dynamically controlled by the serving system.

A related direction is training models to speculate well together. Normally you train a large model and separately find a drafter, which may disagree with it often. Matryoshka model suites instead nest smaller models inside the larger one, so a small, medium and large model are literally subsets of the same network, trained together. The drafter and the verifier are aligned because they were designed as a family rather than introduced after the fact.

Once again, serving requirements are reaching back into training.


## Long-Context Serving

As contexts stretch toward a million tokens, the cache becomes enormous.

Decode Context Parallelism attacks this directly. Ordinary tensor parallelism divides computation by attention heads or model dimensions — but newer attention architectures have very few distinct key-value heads, which can leave duplicated KV state on every GPU. Decode Context Parallelism instead divides the context itself. Given a two-hundred-thousand-token conversation and four GPUs, each GPU owns one section of the history rather than a redundant copy of all of it, and attention combines information across devices when needed.

That costs communication, and it dramatically reduces the KV memory each GPU must hold.

The result is worth noticing carefully, because the individual model operation may not get much faster — but the server fits far more concurrent requests, so total system throughput rises enormously.

That distinction is essential, and it is where most benchmark confusion comes from.


## Cerebras and Where Weights Live

Cerebras illustrates the hardware side of the same memory-movement problem.

Conventional accelerators use relatively small chips attached to large external high-bandwidth memory, and the compute units repeatedly read weights and state from it. Cerebras instead occupies an enormous portion of a silicon wafer and carries a very large quantity of fast on-chip SRAM, so far more of the model's working data stays physically close to the compute.

Why that matters: if generating a token requires pulling enormous quantities of weights through a memory interface, then no matter how fast the arithmetic is, the token cannot be produced until the weights arrive. Keep the weights near the processors and that bottleneck shrinks. This is why wafer-scale inference can hit extremely high interactive token rates.

The lesson is not that everyone will build wafer-scale processors. It is that the physical placement of weights has become part of the serving algorithm. Where do parameters live? How often do they move? Across what interconnect? At what precision? Are inactive experts sitting in host memory? Can weights be pulled from another tier?

Mixture of experts makes this vivid. If a model has hundreds of experts but needs only a few per token, keeping every expert permanently in expensive GPU memory is wasteful. ExactMoE explores keeping some expert weights outside the GPU and bringing them in on demand — frequently used experts stay resident, rarer ones live in host memory and transfer in when required. The benefit is much lower GPU memory use. The danger is transfer latency: if the router asks for an absent expert, a large amount of weight data has to move before computation continues.

So minimizing GPU memory is not the same as maximizing serving quality. You have to weigh footprint against throughput and latency together.

DeaMoE asks a different question: what if many experts share common structure? Separate the shared components from the expert-specific ones and the system stops reloading the same information for every expert. Again the theme holds — the unit that matters is not operations, but useful computation per byte transferred.


## Recurrent Depth

Traditional Transformers give every layer its own parameters. Recurrent-depth architectures reuse the same block many times — instead of twenty-four unique floors, a smaller number of modules that the representation passes through repeatedly. A system like RecurrentGPT can use far fewer unique parameters while still doing substantial computation.

Why does that help inference? Because unique parameters have to be stored and moved. Reusing the same weights several times shrinks the parameter footprint even when total computation is similar. On hardware where bandwidth is expensive relative to arithmetic, that is a very attractive trade: rather than continuously loading new weights, do more work with weights you already hold.

But it creates its own serving problem. What happens to the cache? If every recurrent pass stores its own keys and values, the cache can grow substantially even though unique weights shrank — and you can lose through recurrent KV storage most of what you saved on parameters.

So recurrent depth needs more than weight sharing. It needs a serving-aware state design: share some recurrent KV, compress it, replace some attention-based steps with fixed-size recurrent state, or let only selected loops retain full cache entries. Recurrent depth cannot be judged by parameter count alone. Its real value depends on what happens to persistent inference state.


## The Full-bandwidth Transformer

One of the most interesting architecture papers of the month starts from an odd observation.

In a normal autoregressive Transformer, an enormous amount of internal computation happens before the model picks a token — and then all of that richness collapses into one discrete token, which is what the next step receives.

A Full-bandwidth Transformer also feeds part of the previous step's hidden representation directly into the next step. The model does not communicate with its future self only through words; it can pass forward a richer internal representation as well.

Think about solving a hard problem in your head. If you had to fully verbalize your entire mental state after every small reasoning step and then erase everything except those words, that would be enormously wasteful. Human reasoning does not appear to work that way — we hold continuous internal representations carrying far more than whatever sentence we say aloud.

The emitted token still matters. But a latent state continues forward alongside it. The reported experiments suggest better training efficiency at very little generation-time cost.

And it introduces yet another serving object. The server may now have to carry a persistent latent feedback state between tokens, on top of everything else — exact recent memory, compressed recurrent memory, latent reasoning state, retrieved memory, speculative state, tool state, session state. Managing a portfolio of state types is becoming the job.


## Sessions, Not Requests

That brings us to GPT-Live and continuous inference.

Traditional APIs encourage us to imagine interaction as independent requests: a prompt arrives, the server answers, the request ends. An always-on voice agent is not shaped like that at all. It may be listening while speaking. The user can interrupt. Audio arrives continuously. The model may call another model or tool, hold state for hours, and migrate to different hardware without the conversation restarting.

So the abstraction changes. Instead of requests, sessions. Instead of request latency, continuous real-time deadlines. Instead of rebuilding state from a transcript, preserving active model state. Instead of asking how many requests per second a server handles, asking how many simultaneous live conversations the infrastructure can hold.

Agent serving starts to look less like a web server answering calls and more like an operating system managing long-running processes. A live agent has state. It occupies resources. It may migrate, need more compute temporarily, delegate work, compact its memory, suspend, resume, or fork. Those are operating-system concepts.

Which leads to another intriguing idea: cross-model KV transfer. Suppose a conversation starts on a small, cheap model, most turns are easy, and then a genuinely hard question arrives. Normally the larger model would have to reread the entire conversation to build its own cache — expensive at a hundred thousand tokens. Cross-model KV transfer tries instead to translate the small model's internal state into the large model's format.

If that becomes reliable, routing changes character. You could start sessions cheaply, escalate only when necessary, keep most of the computational history, and drop back down afterwards. That is stateful model routing rather than picking a model per request.

Now combine everything. A conversation begins on a small model. Exact KV covers recent history, compressed recurrent state covers older history, and a shared cache holds the system prompt across thousands of sessions. When something hard arrives, state moves to a larger model, which decodes speculatively with a nested drafter whose depth varies with cluster load. The model uses sparse experts, placed near the compute that needs them. Prefill runs on one group of hardware, decode on another, and long contexts are split across GPUs by position — and the session can migrate between workers without losing continuity.

At that point, what exactly is "the model"? The intelligence is no longer identifiable with one checkpoint. The effective system is the checkpoint plus the drafter, the cache policy, the quantization format, the expert placement, the state-transfer mechanism, the scheduler, and the topology of the cluster.


## Quantization

Quantization reduces the number of bits used for weights or activations, and August systems increasingly use four-bit formats such as NVFP4.

The obvious benefit is that the model occupies less memory. For inference, the more important benefit is that fewer bits means fewer bytes have to move. If a weight takes four bits instead of sixteen, you can push far more weights through the same memory interface in the same time — which is exactly what a bandwidth-limited decode needs.

Done badly it costs accuracy, so modern quantization is not simply shrinking numbers; the calibration process has to preserve the most important information while cutting representation cost. The serving checkpoint becomes a deliberately engineered artifact rather than a generic compressed copy.

It gets powerful in combination with speculation. Meta's Muse Glimmer is a dense multimodal model built to run on a high-end consumer GPU: quantize the target aggressively, pair it with a separate speculative model, and the reported setup generates above two hundred tokens per second for a single user on an RTX 5090. Very high interactive rates are not a datacenter privilege any more.

But do not attribute that number to the base architecture. It is a system result. Remove the drafter, change the precision, change the runtime or the workload, and the number moves.


## Reading Benchmark Claims

Which brings us to interpreting performance claims. When somebody advertises an enormous tokens-per-second figure, ask six questions.

Is that output tokens or total processed tokens? Is it one user or aggregated across many? What is the concurrency? What are the input and output lengths? What hardware and precision? And what happened to time to first token and per-user latency?

Those answers can invert the meaning of a benchmark. In August, OpenAI and Cerebras demonstrated GPT-5.6 Sol Ultrafast at roughly seven hundred and fifty generated tokens per second — an interactive, single-stream claim. Meanwhile vLLM demonstrated Qwen3.5 at more than twenty-five thousand total tokens per second per GPU under very high concurrency — a fleet-throughput claim. Both are impressive. They answer different questions, and each individual user on the second system experiences only moderate speed.

So a single number is never enough. You need the Pareto frontier: the tradeoff surface across per-user speed, total throughput, memory, quality, power and cost, where improving one thing costs you another. The best serving reports show several operating points — tuned for fast individual generation at low concurrency, trading responsiveness for aggregate throughput at high concurrency.

If you are building an interactive coding agent, the first number is what you care about. If you run a batch platform, the second is.


## What It All Adds Up To

Three architectural movements are happening at once, and each attacks a different kind of waste.

Sparse computation says we should not activate every parameter for every token. Compressed or hierarchical state says we should not store every detail of every previous token forever. Amortized generation says we should not need one full expensive model step per accepted output token.

All three push complexity into the serving layer. Sparse experts need routing and placement. Recurrent state needs lifecycle management. Speculation needs drafting, verification, acceptance logic and adaptive scheduling. That is why serving has moved to the center of architecture research. The interesting question is no longer which network achieves the best loss, but which combination of architecture and runtime achieves the most intelligence per dollar, per watt, per byte moved, and per millisecond the user actually perceives.

The durable lessons are these. Full attention does not disappear, but we use less of it, invoking it periodically when precise global access is worth paying for. Sparse mixture of experts is now inseparable from systems engineering — the hard part is placing experts and moving their weights, not choosing them. Prefill and decode are separate workloads. Speculation becomes adaptive, tuned to confidence and load rather than fixed. And inference state is a first-class computational resource that has to be managed explicitly.

One sentence for the whole month: the frontier of inference is moving from optimizing calculations to optimizing movement and state.

The winning systems ask which parameters truly need to activate, which bytes truly need to move, which history truly needs to stay exact, which computation can be reused, and how many accepted tokens can be extracted from each expensive pass.

That is why Qwen3.8 is instructive — not because it has trillions of parameters, but because sparse experts, recurrent state, occasional full attention, low precision, speculative generation, typed caches and split prefill and decode are coordinated into one working system. It is why the Cerebras result matters: hardware redesigned around the memory behavior of autoregressive inference. And it is why Full-bandwidth Transformers and recurrent-depth models deserve attention despite being less mature, because they question two assumptions we rarely examine — that information between steps should travel as discrete tokens, and that more depth requires more unique weights.

So when you evaluate a future architecture, do not just ask whether it has a cleverer attention mechanism. Ask what happens when ten thousand concurrent users sit behind it. What state does each one create, how fast does it grow, and where does it live? Can prefixes be shared? Can sessions migrate? Can the weights stay close to the compute, and the inactive ones somewhere cheaper? Can one expensive invocation yield several accepted tokens?

That is the difference between an architecture that looks impressive in a paper and one that becomes a usable platform.

Because the model no longer ends at the edge of the neural network. The cache is part of the model. The scheduler is part of the model. The drafter, the precision format, the memory hierarchy, and increasingly the topology of the cluster are all part of the effective architecture.

That is the shift worth watching.
