# Thinking in the Dark
### A technical lecture on recurrent-depth models
*Approx. 45–55 minutes spoken. Written to be listened to — the math is spoken aloud rather than rendered, so it should read cleanly through a text-to-speech engine.*

---

## Cold open

Welcome back. Today I want to talk about one of the stranger, and I think more underrated, ideas in modern deep learning: the idea that a neural network can *think* by running the same computation over and over, in the dark, inside its own latent space, before it ever emits a single word.

Here's the framing I want you to hold onto for the whole lecture. There are basically three ways you can spend more compute on a language model. The first is the one everyone knows: make the model *bigger*. More parameters, more data, more pretraining FLOPs. That's the Kaplan-style scaling story, and it's expensive and it's hitting diminishing returns. The second axis is newer and it's what the reasoning-model wave of the last couple of years has been about: spend more compute *at test time* by having the model write out a long chain of thought. It talks to itself on the page, token after token, and somewhere in that monologue the answer falls out.

Recurrent-depth models are a bet on a *third* axis. Instead of making the model wider, and instead of making it talk more, you make it *iterate*. You take a block of layers and you run it in a loop — five times, thirty times, a hundred times — refining a hidden state each pass, and only at the end do you decode that state into a token. The reasoning happens in a continuous vector space, not in words. It's silent. And the depth of that reasoning — the number of loops — is something you can crank up or down at test time, per token, without retraining anything.

That's the whole idea in one breath. The rest of this lecture is about *why* that works, *how* you actually train such a thing without it blowing up, the beautiful mathematics of differentiating through a fixed point, and the genuinely weird things that emerge when you scale it up — models that trace orbits in latent space to do arithmetic, that spend more "thinking time" on a morally ambiguous question than on a high-school algebra problem, entirely on their own.

Let's build it from the ground up.

---

## Part 1 — What depth buys you, and why you'd ever share weights

Start with the basic question: what is *depth* doing in a transformer? Each layer reads the residual stream, does a little attention, does a little feedforward computation, and writes something back. Stacking layers lets the model compose operations — early layers resolve tokens into concepts, middle layers do the heavy relational work, late layers shape things into a prediction. More depth means longer chains of composed computation, which means the model can express more complex functions and, loosely, run "more steps" of whatever algorithm it has implicitly learned.

Now here's the key observation that the whole field rests on. In a standard transformer, every one of those layers has *its own* parameters. Layer twelve and layer thirteen are different functions with different weights. But there's no law of nature that says they have to be. What if layers twelve through twenty were all *the same function*, applied repeatedly? You'd be tying the weights across depth. You'd have one block of parameters, and "depth" would just be how many times you chose to apply it.

The moment you do that, something important decouples. In an ordinary network, effective depth and parameter count are welded together — sixty layers means sixty layers' worth of weights. But with a weight-tied, looped block, those two quantities come apart. You can have a tiny number of *actual* parameters and an enormous *effective* depth, because the same parameters get reused on every pass.

The paper at the center of this lecture has a lovely way of talking about this. They distinguish the model's real parameters from its *materialized* parameters — the parameter count you'd need in a normal feedforward network to do the same amount of computation. Their big model has about three and a half billion real parameters, but when you run the recurrence thirty-two times, it chews through FLOPs comparable to what a thirty-two-billion-parameter dense transformer would burn, and at higher iteration counts it reaches compute loads equivalent to a fifty-billion-parameter model. Same weights. Far more thinking.

And notice the inversion this sets up against mixture-of-experts models, which we'll come back to at the end. A mixture-of-experts model is *parameter-heavy* but *compute-light* — it stores a huge number of weights and activates only a few per token. A recurrent-depth model is the mirror image: *compute-heavy* but *parameter-light*. It stores few weights and runs them many times. Those two designs are duals of each other, and that duality is going to matter for hardware.

So weight-tying in depth is the foundational move. Everything else is about making it actually train, actually converge, and actually get smarter when you give it more loops.

---

## Part 2 — The ancestors

This idea is old. It's one of those ideas that gets rediscovered every decade — as recurrent neural networks, as diffusion models, as looped transformers. Let me walk through the direct ancestors, because each one contributes a piece of machinery we'll need.

**First: cross-layer parameter sharing.** The cleanest early example in the transformer world is ALBERT, which simply shared one layer's parameters across all the layers of a BERT-style encoder. The motivation there was mostly parameter efficiency — a smaller model that performs comparably. But the number of layers was fixed. It wasn't a knob you turned at test time; it was just a way to compress. Still, it's the proof of concept that weight-tying across depth doesn't destroy a transformer.

**Second, and more important: the Universal Transformer, with Adaptive Computation Time.** This is where the recurrence becomes genuinely dynamic. Take a single transformer block, share its weights, and apply it repeatedly — recurrent in depth. To each position, on each step, you add a timestep embedding alongside the usual positional embedding, so the block knows *which iteration* it's on and *where* it is in the sequence.

But the clever part is the halting mechanism, which comes from Alex Graves's Adaptive Computation Time. The idea: not every token needs the same amount of processing. So at each position, on each step, the model emits a little scalar halting probability through a sigmoid. You accumulate those probabilities across steps, and as soon as the running sum at a given position crosses one minus a small epsilon, you *stop* updating that position — you freeze its state. The leftover probability mass, the "remainder," is used to weight the final contribution so the whole thing stays differentiable. And you add a "ponder cost" to the loss, a penalty proportional to how many steps each position took, so the model is pressured not to think forever. The dream behind the Universal Transformer was explicit and ambitious: by recurring in depth with dynamic halting, you're reaching toward a *Turing-complete* machine, something that can in principle run arbitrary-length computations.

Hold onto two ideas from this: *per-position adaptive depth*, and *a halting criterion*. Both come back, in a much simpler form, at the end.

**Third: the "deep thinking" literature.** This is a line of work — Schwarzschild, Bansal, and collaborators — that asked a sharp question. If you train a recurrent network on *easy* instances of a problem, using a small number of iterations, can it solve *harder* instances at test time just by iterating *more*? Can it extrapolate along the compute axis? And the answer, with the right setup, is yes — and they figured out *what* the right setup is. Two ingredients turned out to be essential.

One: **input injection.** You don't just feed the input in at the start and let the recurrence run free. You re-inject the embedded input into the block on *every single iteration*. We'll see in a moment why that's not optional — it's what makes the iteration mathematically well-behaved.

Two: **randomized unrolling.** During training you don't use a fixed number of iterations. You sample the iteration count randomly for each example. This teaches the network to make progress at *any* depth, rather than memorizing a fixed-length computation, which is exactly what lets it extrapolate to more steps later.

And there's a third property that this community identified, which is going to be a recurring character: **path independence**, in the sense of Anil and collaborators. A path-independent recurrent model converges to the same answer *regardless of where it started* — regardless of the random initial state. Different starting points, same destination. That's a stability property, and it turns out you can encourage it precisely through random initialization plus input injection.

So the ancestors hand us four tools: weight-tying, dynamic per-position halting, input injection every step, and randomized unrolling for extrapolation. Now let's get to the mathematically deepest ancestor, because it's where the most beautiful trick lives.

---

## Part 3 — The fixed-point view, and the implicit-function-theorem magic

Here's a thought. If you're going to apply the same function over and over — call it *f*, taking the current hidden state and the input — what happens if you just... keep going? Forever? In many cases the state stops moving. It settles. You reach a *fixed point*: a state, call it *z-star*, where applying *f* to it gives you back *z-star* again. The function maps it to itself. Nothing changes.

Deep Equilibrium Models — this is Bai, Kolter, and Koltun — take that observation and run with it about as far as it can go. Their move is radical: don't pick a number of iterations at all. Instead, *define* the network's output to be the fixed point of *f*. The output is the solution to the equation "*z* equals *f* of *z* and *x*." An infinitely deep, weight-tied network, represented not by unrolling but by the equilibrium it converges to.

The forward pass, then, isn't "run thirty steps." It's "find the root." And you can use any black-box root-finding solver to do it — Broyden's method, Anderson acceleration, whatever converges fastest. You don't care how many solver steps it takes; you just want the equilibrium.

But now you hit the obvious problem, and this is the part I really want you to sit with, because the solution is gorgeous. *How do you backpropagate through this?* If the forward pass was a black-box solver that took some unknown number of iterations, you absolutely do not want to store every one of those iterations and backprop through all of them. That would cost memory proportional to the number of solver steps, which is exactly what we're trying to avoid.

The escape is the implicit function theorem. Watch.

At the fixed point, we have the identity: *z-star* equals *f* of *z-star* and *x*, where the model's weights are *theta*. This holds *at the solution*. Now, differentiate both sides with respect to *theta*, treating *z-star* as an implicit function of *theta* — because it is; move the weights, the equilibrium moves. By the chain rule, the change in *z-star* equals the partial of *f* with respect to its first argument — that's the Jacobian of *f* at the fixed point, call it *J* — times the change in *z-star*, plus the partial of *f* with respect to *theta* directly.

Rearrange. Bring the *J* term to the left. You get: the quantity "identity matrix minus *J*," times the derivative of *z-star* with respect to *theta*, equals the partial of *f* with respect to *theta*. So the derivative of the fixed point with respect to the weights is "identity minus *J*," *inverted*, times that partial.

And therefore the gradient of your loss with respect to the weights is: the gradient of the loss with respect to *z-star*, times "identity minus *J*" inverse, times the partial of *f* with respect to *theta*.

Stare at that for a second. The entire dependence on *how you got to the fixed point* has vanished. The solver's trajectory is gone. The gradient depends *only* on quantities evaluated *at the equilibrium itself* — the Jacobian there, and the partial derivatives there. You never unroll. Memory is constant — order one — in the number of solver iterations. That's the headline result of Deep Equilibrium Models: implicit differentiation gives you constant-memory backprop through an effectively infinite-depth network.

Now, the one piece that looks scary is that inverse Jacobian, "identity minus *J*" inverse. You're not going to form that matrix explicitly — it's enormous. But you don't have to. Look at the vector you actually need: the loss gradient times that inverse. Call it *u*. Then *u* satisfies its own equation: *u* equals the loss gradient with respect to *z-star*, plus *u* times *J*. And *that* is itself a fixed-point equation. You can solve it with the *same* kind of iteration you used on the forward pass, using only vector-Jacobian products — which automatic differentiation gives you cheaply, without ever materializing *J*. So you solve one fixed point going forward, and a second, linear fixed point coming back. Elegant, and entirely practical.

So why isn't every model just a DEQ? Because that inverse is exactly where the trouble lives. If the Jacobian *J* has eigenvalues near one, then "identity minus *J*" is nearly singular — ill-conditioned — and your backward solve becomes unstable and slow. The forward root-find can also fail to converge if *f* isn't contractive enough. People patch this with Jacobian regularization — literally adding a penalty on the norm of *J* to the training loss to keep the dynamics tame — and with careful solver choices. But it's *finicky*. Driving a big language model to a clean fixed point on every forward pass, stably, at scale, is hard. And that fragility is the opening that the modern approach walks through. Keep DEQ in mind as the *clean mathematical ideal* — the elegant limit — and remember that the scalable system we're about to build deliberately *doesn't* take that limit.

---

## Part 4 — How you actually get gradients through a loop

Before we assemble the modern model, let's lay out the menu of ways to differentiate through a recurrence, because the central design decision is *which one you pick*.

**Option one: full backpropagation through the unrolled computation.** Just treat the *r* iterations as a deep network and backprop through all of them. This is backpropagation through time, the classic RNN approach, repurposed for depth. It's exact. And it costs memory proportional to *r*, because you have to cache the activations of every iteration for the backward pass. If you want to iterate a hundred times, you pay for a hundred layers' worth of stored activations. For a big model with a heavy-tailed distribution of iteration counts, that's a non-starter — your peak memory is set by your worst-case longest unroll.

**Option two: gradient checkpointing.** Don't store every iteration's activations; store a few checkpoints and *recompute* the rest during the backward pass. This trades compute for memory — you do the forward work twice — but it lets you fit longer unrolls. It softens the memory problem without solving the fundamental scaling-with-*r* issue.

**Option three: the implicit gradient we just derived.** Constant memory, but it assumes you've actually reached a fixed point and inherits all the conditioning headaches. The clean ideal.

**Option four: truncated backpropagation.** Run the recurrence forward for the full *r* steps — that part is cheap, it's just inference — but only backpropagate through the *last k* of them, for some small fixed *k*. You throw away the gradient signal from the early iterations. This is biased — you're not computing the true gradient — but in practice the recent steps carry most of the useful signal, and crucially your backward memory and compute become *independent of r*. You can sample a wild, heavy-tailed number of forward iterations and your training cost per step doesn't budge.

That fourth option — truncated backprop through depth — is the pragmatic choice the modern model makes, and it's worth appreciating *why* it's the right call. DEQ solves what you might call the "direct" problem — it commits to the fixed point and pays for that commitment in stability. Diffusion models, another iterative paradigm, train against a surrogate objective. Truncated unrolling is a third path: don't force a fixed point, don't use a surrogate, just unroll a random amount and learn through the tail end. The bet — and it's an empirical bet that pays off — is that this is the *scalable* option, the one that survives contact with a real billion-parameter training run on a real supercomputer.

---

## Part 5 — The modern synthesis: latent recurrent depth at scale

Now we build the actual model. This is the Geiping and collaborators work from early 2025 — "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" — and the model they release is called Huginn, after one of Odin's ravens, the one whose name means "thought." I'll describe it piece by piece, and I'll flag *why* each choice is made, because almost every design decision is a scar from something that broke.

### The macroscopic shape: prelude, core, coda

The model has three functional groups. First, the **prelude** — a couple of ordinary transformer layers whose job is to take the input tokens and embed them into the latent space. Call its output *e*, the embedded input. Second, the **core recurrent block** — this is the engine, the part that loops. Third, the **coda** — a couple of layers at the end that take the final latent state and un-embed it, projecting back out to vocabulary logits, the next-token prediction.

The forward pass goes like this, in words. Run the prelude on the input to get the embedding *e*. Initialize a latent state — call it *s-naught* — by drawing it from a Gaussian, random noise. Then loop: the *i*-th state is the core block applied to two things, the embedding *e* and the previous state. Do that *r* times. Take the final state, run it through the coda, and out come your token probabilities.

They summarize the architecture with a triplet of numbers — layers in the prelude, layers in the core, layers in the coda. The big model is two, four, two. Two prelude layers, four in the recurrent core, two in the coda. That's *eight real layers* of weights. But run the core thirty-two times and the effective depth is: two, plus four times thirty-two, plus two — a hundred and thirty-two layers. From eight layers' worth of parameters, you've materialized a hundred-and-thirty-two-layer computation, deeper than essentially any fixed-depth transformer anyone trains. The hidden dimension is five thousand two hundred and eighty, which works out to fifty-five attention heads of size ninety-six. About one and a half billion parameters live in the non-recurrent prelude and head, about one and a half billion in the recurrent core, and another half-billion in the tied input embeddings.

### Why you inject the input every single step

This is the most conceptually important design choice, so let me give it real time. Why feed *e* into the core on *every* iteration, instead of just once at the start?

The paper's intuition is a gradient-descent analogy, and it's exactly right. Imagine the iterative process you'd most want the model to be able to imitate: gradient descent minimizing some function that depends on both a variable and the *data*. You start from a random point, and you repeatedly take a step — and that step is the gradient of the objective, which *depends on the data*. The data shows up in *every* update. It has to. If the data only entered at initialization and then you iterated a data-independent map, you couldn't be doing data-dependent optimization at all.

The same logic applies here. If you only set the initial state equal to *e* and then iterated a block that *didn't* see *e* again, the final answer would depend only on the boundary condition — the starting point — and the fixed map. The block couldn't represent something like gradient descent on a data-dependent objective, because, formally, it couldn't be the kind of monotone operator that such optimization requires. The recurrence wouldn't be stable in the relevant sense. So you re-inject *e* every step. Mechanically, the core starts with an adapter — a learned matrix that takes the *concatenation* of the current state and the embedding, both of dimension *h*, and maps that two-*h*-dimensional vector back down to *h*. At small scale you can get away with just *adding* the embedding back in; at scale, *concatenation* works better.

### Why you start from random noise — path independence again

Why initialize *s-naught* as random Gaussian noise instead of something deterministic? Because random initialization, combined with input injection, is what drives the model toward *path independence* — convergence to a steady behavior that's independent of where you started. During training, the random start forces the model to learn dynamics that work from *anywhere*, not from one privileged initialization. And the payoff is verified empirically at the end: if you re-run the trained model from several different random starting states, it traces *similar trajectories* and lands in the same attractors. Same fixed points, same orbits, regardless of the seed. That robustness is not decorative — it's what makes "just iterate more at test time" a sane thing to do.

There's a nice aside in the paper here: isn't this basically a diffusion model? Random initial state, iterative refinement — it rhymes. They tried making it *more* diffusion-like, injecting fresh noise at every step, and also tried letting the block condition on the step index the way diffusion models condition on the timestep. Both *hurt*. The step-conditioning in particular broke path independence and wrecked the model's ability to extrapolate to more iterations than it saw in training — which makes sense, because if the block knows "I am on step seventeen of thirty-two," it can specialize to that schedule rather than learning a step-agnostic operator you can run as long as you like.

### The norm placement and the initialization — the part that kept breaking

Here is where I want to be honest about how fragile this is at scale, because the paper is unusually candid about its failures, and the failures are instructive.

The layers inside each block use a specific normalization layout they call a "sandwich" — RMSNorm placed both before *and* after each sub-layer, attention and feedforward alike. Concretely, you normalize the input to attention, add the residual, normalize again; then normalize the input to the feedforward, add the residual, normalize again. At small scale, this barely matters — pre-norm, post-norm, sandwich, they all train fine. At scale, it's the difference between a working model and a dead one.

What goes wrong? **Representation collapse.** Their first big run used a more conventional setup, and it stalled almost immediately. When they looked at why, they found that the correlation between the hidden states of *different tokens* in the sequence shot up toward one. The model was predicting *the same hidden state for every token*. The sequence had homogenized into mush. And the recurrence was the culprit — every iteration of the block nudged the token representations closer together, and over many iterations they collapsed completely. The loop *amplifies* this pathology in a way a fixed-depth network never would.

They tried to fix it — added an embedding scale factor, switched to a learned adapter, went back to pre-norm — and the second run *looked* fixed at first. Token correlation spiked but then recovered. But it landed in a different trap: that model couldn't *use* test-time compute at all. Its validation perplexity was identical whether you ran the core once or thirty-two times. It had learned, early, to *ignore the incoming state* — to treat the recurrence as decorative. A local minimum where the loop does nothing.

Only the third configuration worked: back to the sandwich norm, plus a careful initialization, plus cutting the peak learning rate hard — roughly an order of magnitude, down to about four-times-ten-to-the-minus-five. On the initialization: they use a scheme that sets the weight variance to two-fifths over the hidden size, draws everything from a truncated normal, and critically sets the *output projection* layers to a much *smaller* variance — scaled by the number of *effective* layers, which is over a hundred. Small output projections mean each layer makes a gentle contribution, which keeps a very deep unrolled computation from exploding. With that combination, the third run started clean, never approached token collapse, and actually improved as you gave it more iterations. They trained it for seven hundred and fifty billion more tokens without a single loss spike.

The lesson I want you to take from this isn't the specific recipe — it's that recurrence is a *stability amplifier*. Any small pathology in the dynamics — a tendency to collapse, a tendency to ignore inputs — gets compounded across iterations. Designing one of these models is, to a large degree, the art of designing a *stable* iterated operator. The norm placement and the small output-projection initialization are both, at bottom, about keeping the per-step map well-behaved enough to apply a hundred times.

### The training objective — sampling how long to think

Now, how do you train across a *range* of iteration counts so the model works at any depth? You make the loss an expectation. You minimize the *expected* next-token loss, where the expectation is over both your data and a *random iteration count r*, drawn fresh for each sequence from some distribution.

What distribution? They use a **log-normal Poisson**. The construction: pick a target mean number of iterations — for the big model it's thirty-two — then sample a log-normal variable, exponentiate it, and use that as the rate of a Poisson, plus one. With their variance setting, this gives a distribution that *most often* samples *fewer* than thirty-two iterations — the mode is around twenty-four, the median around twenty-nine — but has a long, heavy *tail* that occasionally demands many more. Why that shape? The bulk of cheap samples keeps average training cost down. The heavy tail occasionally exposes the model to deep unrolls, which is what teaches it to keep making progress far beyond the typical depth — that's the extrapolation property from the deep-thinking literature, baked directly into the sampling distribution.

And here's where truncated backprop earns its keep. They run the full sampled *r* iterations forward, but they only backpropagate through the **last eight**. *k* equals eight, fixed. Because backward memory and compute depend only on *k* and not on *r*, that heavy Poisson tail is *free* on the backward pass — a sample that happens to demand ninety forward iterations costs the same to train on as one that demands ten. One subtlety: even though gradients only flow through the last eight core iterations, the *prelude* still gets a gradient on *every* step, because its output *e* is injected into the core at every iteration and therefore participates in all of them. So the input embedding is trained against the full depth even while the core's backward pass stays cheap.

There's also a distributed-systems wrinkle that I love because it shows how the math meets the metal. If every worker in a data-parallel cluster sampled its *own* random *r*, then on each step all the fast workers would sit idle waiting for the unlucky one that drew a huge iteration count to finish its backward pass. So they use **locked-step sampling**: one *r* per micro-batch, synchronized across all workers, so everyone unrolls the same depth and nobody stalls. It's a small compromise on faithfully modeling the expectation, in exchange for not wasting thousands of GPUs.

### The scale and the hardware, because it's the point

This was trained on Frontier, the Oak Ridge supercomputer, on AMD MI250X GPUs — and the hardware story is not incidental, it's *the whole pitch* for this architecture. Remember: recurrent-depth models do many FLOPs per parameter. The weights are small and reused. That means the model is small enough to train with *pure data parallelism* — no tensor parallelism, no model sharding across devices — because the weights fit comfortably and you're just doing a lot of compute on them. Pure data parallelism means *minimal communication between GPUs*. And on a cluster with slower interconnects, communication is exactly what kills you. So an architecture that is compute-heavy and parameter-light is *precisely* the architecture you want when your bottleneck is the network between accelerators rather than the accelerators themselves. They ran on up to four thousand and ninety-six GPUs, with global batches around sixteen million tokens, pushing on the order of a million tokens a second, and trained the final model on about eight hundred billion tokens — most of it scheduled in twelve-hour chunks across December of 2024.

This is the deeper argument for the third axis. It's not only "thinking in latent space is nice." It's that the resulting compute profile maps beautifully onto hardware where bandwidth, not raw FLOPs, is the constraint.

---

## Part 6 — The free lunch at inference

Here's the genuinely delightful part. Because the model was trained to operate at a *range* of depths, and because it's path-independent, a whole menu of capabilities that normally require dedicated engineering just... fall out. Zero-shot. No extra training. Let me go through them.

**Test-time compute scaling.** The headline. You give the model a hard problem, you let it iterate more, and it gets better. On reasoning benchmarks, accuracy climbs with iteration count — and the climb is *task-dependent*, which is the beautiful part. Easy tasks saturate fast: something like HellaSwag, a commonsense-completion task, basically maxes out by eight iterations. Harder tasks keep improving much longer — grade-school math word problems keep getting better as you add compute. And here's a really telling result: the more *in-context examples* you give it, the more iterations it *chooses* to use before saturating. With no examples, it plateaus around eight to twelve iterations; with one example, around twenty; with twenty-five or fifty examples in context, it keeps using compute out to thirty-two iterations. It's spending more thinking time to actually digest the additional context. The three-and-a-half-billion-parameter model, given enough iterations, reaches reasoning performance competitive with the compute of a fifty-billion-parameter dense model.

**Per-token adaptive compute, for free.** This is the Universal Transformer's dream, achieved without any of the machinery. Remember adaptive computation time — the halting units, the ponder cost, the accumulator? You don't need any of it. Here's their entire halting criterion: at each iteration, look at the output distribution the coda *would* produce. Compare it to the previous iteration's output distribution using KL divergence. When the KL between successive steps drops below a small threshold — they use five-times-ten-to-the-minus-four — the state has stopped meaningfully changing, so you stop, decode the token, and move on. That's it. No exit heads trained at every layer, no auxiliary losses. And when you measure how many steps different tokens take, the structure is striking: on high-school math questions the model exits quickly; on moral-scenario questions it takes several steps *more* on average before it's confident. The model is spending more silent deliberation on the harder, fuzzier questions, entirely on its own, even though it was only ever trained with a *single* fixed depth per whole sequence. Per-token variation was never in the training objective. It emerged.

Now, you'd think per-token early exit would break the KV cache — that's the standard objection. If token thirty exits after five iterations but token thirty-one wants to run twenty iterations, then when token thirty-one's attention reaches back to token thirty, the deep KV entries it wants simply *don't exist* — token thirty stopped early. The fix is clean: you just attend to the *deepest available* cached state for each previous token. And the reason this works is subtle and specific to the architecture — because every recurrent step uses the *same* key and value projection matrices, the cache entries from different depths are mutually compatible. They live in the same space. A token computed to depth five and a token computed to depth twenty produce keys and values that "match" well enough to attend across. The weight-tying that defines the architecture is exactly what makes the cache forgiving.

**KV-cache sharing, for free.** Same flavor of trick. Normally, sharing a KV cache across layers is something you have to train for from scratch. Here, because all recurrent steps share projections, you can just set a fixed cache budget — say sixteen slots — and have iteration *i* read and write slot *i* modulo sixteen. The seventeenth iteration overwrites the first iteration's entry, and so on. Cuts memory, barely touches quality.

**Self-speculative decoding, for free.** Speculative decoding normally needs a separate, smaller draft model to propose tokens that the big model then verifies. Here you don't need one. You *draft* with a few iterations — cheap, fast — and then *verify* with more iterations. The model is its own draft model, just run shallower. And the states you computed while drafting aren't wasted; they're the early iterations of the verification pass, so you reuse them. The single architecture gives you the draft-and-verify pair for nothing.

**Continuous chain-of-thought, for free.** And this one closes a loop with the reasoning-model world. Instead of re-initializing *s-naught* to fresh random noise for every new token, you can *warm-start* it with the *final* state from the *previous* token. Now the latent computation carries forward across tokens — the model's "thoughts" about token thirty inform its thinking about token thirty-one, building a computational graph deeper than any single token's iteration count. This is the same idea as the "continuous thought" line of work — Coconut and relatives — where you feed a model's last hidden state back in as the input for the next reasoning step. The interesting observation is what *distinguishes* the approaches: that other line *finetunes existing fixed-depth transformers* on chain-of-thought data to accept their own hidden states as input, effectively retrofitting a limited depth-recurrence onto a model that wasn't born with it. This work pretrains for recurrence from scratch. Same destination — reasoning in continuous latent space — approached from opposite directions: build it in, versus bolt it on. And the from-scratch version needs no chain-of-thought data at all.

---

## Part 7 — What's actually happening in there

So the model is silently iterating a hidden state and getting smarter. The obvious, slightly unnerving question is: *what is it doing* during all those loops? You can't read it. There's no transcript. But you can *watch the state move*. You track the trajectory of the latent state across iterations and look at its geometry. And what they find, when they project these high-dimensional trajectories down with PCA and plot them, is genuinely surprising, and it emerges *purely from scale and the plain truncated-unrolling objective* — there is nothing in the loss that asks for any of this.

Most tokens do the boring, expected thing: the state spirals in and *converges* to a fixed point. The model thinks for a bit and settles. Fine.

But some tokens **orbit**. The state falls into a closed, periodic loop and circles — it doesn't converge, it cycles. And this isn't noise; the orbits show up consistently on specific *kinds* of tokens. Tokens doing arithmetic. The paper has a lovely concrete example: a grade-school math problem that begins "Claire makes a three-egg omelette," and the token for "three" — the number being operated on — falls into a clean orbit across multiple PCA planes. The interpretation they offer is that these multidimensional orbits might serve the same role as the periodic, circular structures people have found inside fixed-depth transformers trained to do modular arithmetic — except here the periodicity isn't confined to arithmetic; it shows up on tokens like "makes" or "thinks" that *determine the structure of the answer*. The model appears to have discovered, on its own, that cyclic dynamics in a continuous space are a useful way to compute certain things.

And some tokens **slide**. The state doesn't converge and doesn't orbit — it *drifts*, steadily, in a single consistent direction across iterations. The speculation, and it's a good one, is that a slider could be a *counter* — a way for the model to keep track of how many iterations have elapsed, since the magnitude of the drift encodes the step number. Remember, the model was deliberately *denied* explicit step information to preserve path independence. A drifting latent direction would be how it reconstructs a sense of time for itself, when it needs one.

The big-picture contrast I want to draw is with Deep Equilibrium Models. A DEQ has the fixed point *written into its training objective* — it's optimized to converge. This model has *no such prior*. It's only ever trained with truncated unrolling. And yet fixed points emerge, *and* orbits emerge, *and* sliders emerge — a richer repertoire of dynamics than the DEQ objective would ever permit, because the DEQ insists on convergence and these orbits explicitly *don't* converge. Letting go of the fixed-point requirement didn't give you *less* structure. It gave you *more*. The model organizes its computation *spatially*, in the geometry of a high-dimensional space, in ways that have no clean analogue in the linear, one-word-after-another structure of verbalized chain-of-thought. That's the sense in which this might capture reasoning that "doesn't fit into words" — spatial intuition, the rotation of a shape, the kind of thinking that happens before language.

And path independence holds through all of it. Re-initialize from different random states and the *same* orbits, the *same* fixed points, the *same* drifts reappear. The dynamics are a property of the *learned operator and the input*, not of the accident of where you started.

---

## Part 8 — Tradeoffs, tensions, and the honest accounting

Let me not oversell this. Here's the balanced ledger.

**On the positive side.** No bespoke chain-of-thought data — you train on ordinary text and the reasoning ability comes from the architecture, not from curated reasoning demonstrations. Small context windows suffice, because the thinking happens in the hidden state rather than by filling the context with a long monologue, which also means much lower memory at inference than a model generating thousands of reasoning tokens. The capacity to capture non-verbal reasoning. A compute profile — many FLOPs per parameter — that's ideal for bandwidth-limited clusters. And that whole suite of inference capabilities that come for free.

**On the negative side, and these are real.** First, *training fragility* — we spent a whole section on it. The norm placement, the initialization, the learning rate, the input injection: get any of them wrong and you get representation collapse or a model that ignores its own recurrence. This is a less forgiving thing to train than a standard transformer.

Second, *convergence is not guaranteed and arguably not even the goal*. Truncated unrolling doesn't force a fixed point. Sometimes you get orbits, which are fine — even useful — but they complicate any clean story about "the model converging to an answer," and they make a naive convergence-based stopping rule trickier than it sounds.

Third — and to me this is the most important one — *interpretability*. A chain-of-thought model, whatever its flaws, leaves a trace you can *read*. You can audit the reasoning, catch a mistake, notice when it's rationalizing. A recurrent-depth model does its reasoning in an opaque latent space. There is no transcript. We can plot trajectories and label them "orbit" or "slider" after the fact, but we cannot read the model's thinking the way we can read a chain of thought. For a capability whose entire premise is *more reasoning we can't see*, that's a genuine and safety-relevant cost, and it deserves to be named plainly rather than buried.

Fourth, *latency*. The iterations are inherently *sequential* — iteration *i* needs the output of iteration *i*-minus-one. You can't parallelize across depth the way a transformer parallelizes across the token dimension during training. Deep thinking means a long serial dependency chain, and that has wall-clock consequences.

**The honest framing**, which the authors are careful about, is this: recurrent depth is *not a replacement* for the other two axes. It's a *third axis*. It composes with scaling parameters in pretraining, and it composes with scaling verbalized inference. The interesting future is probably *all three together*, not a winner.

---

## Part 9 — The frontier, and where this is heading

Let me end by pointing at the open horizon, because this is very much a live area.

There's a rich body of *theory* on looped and weight-tied transformers framing them as something like *programmable computers* — the looped structure, with the right setup, can express general computation, which is the formal backbone under the Universal Transformer's Turing-completeness dream. There's a growing *survey literature* on "latent reasoning" as its own category, placing recurrent depth alongside the continuous-thought finetuning methods and the latent-space approaches in the big labs' systems. The two cultures — pretrain-for-recurrence versus finetune-an-existing-model-into-recurrence — are converging on the same target from both ends.

On *post-training*, almost everything is open. Can you take one of these models and *compress* the recurrence — distill a thirty-two-iteration behavior into eight? Can you use reinforcement learning, feeding the model problems of graded difficulty so it learns to allocate the right amount of latent thinking to each? Can you *internalize* chain-of-thought data *into* the recurrence, so that reasoning that used to be verbalized becomes silent latent computation? Each of those is a paper waiting to happen.

On *architecture*, there's a natural marriage with efficient attention. Linear-attention variants are fast but limited in how many pairwise comparisons they can make in a single pass — but with recurrent depth, you can just *repeat* the block until all the necessary comparisons have been computed. The loop buys back the expressivity that the efficiency sacrificed. And there's the duality I promised at the very start: mixture-of-experts is parameter-heavy and compute-light; recurrent depth is compute-heavy and parameter-light. They are two ways of decoupling cost from capability, pulling in opposite directions, and the obvious question is what happens when you *combine* them — a model that is both broad in stored knowledge and deep in silent computation.

---

## Closing

So here's the whole arc, one more time, compressed.

Depth in a transformer is composed computation. Tie the weights across depth and effective depth comes unwelded from parameter count — you can be small in weights and enormous in computation. The Universal Transformer made that loop dynamic with per-position halting; the deep-thinking literature found that input injection every step and randomized unrolling let such a model extrapolate to harder problems by simply thinking longer; Deep Equilibrium Models showed the gorgeous mathematical limit, where you define the output as a fixed point and differentiate through it for free with the implicit function theorem, paying only in stability. The modern synthesis — Huginn — declines that fragile limit, chooses truncated backpropagation through the last few iterations, samples a heavy-tailed number of loops during training, and scales the whole thing to billions of parameters on a supercomputer where its compute-heavy, communication-light profile is a feature rather than a cost. And in exchange you get test-time compute scaling, per-token adaptive thinking, self-speculative decoding, and KV-cache tricks all for free — plus a model that, when you watch it think, traces orbits and drifts and fixed points in a latent space, organizing its reasoning *spatially* in a way no chain of thought ever could.

The bet underneath all of it is a simple and slightly profound one. We've spent a few years assuming that for a model to think harder, it has to think *out loud* — to spell its reasoning into tokens. Recurrent-depth models propose that thinking can instead be a *loop*: a quiet, iterated refinement in the dark, with depth as a dial you turn at the moment you need it.

Whether that becomes a pillar of how we build these systems, or a beautiful idea that keeps getting rediscovered every decade and never quite takes over — that's genuinely unsettled. Which is exactly what makes it worth understanding now.

That's the lecture. Thanks for listening.
