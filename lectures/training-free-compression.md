# Compression Without Retraining

### The state of the art in training-free model compression, and where it goes next

*About forty minutes spoken. A survey lecture, built from a literature review of papers up to late 2026. Numbers are rounded and said in words; the exact tables and sources are in the review notes. Where a claim is my own reasoning rather than a published result, I'll say so.*

---

## Why training-free

Suppose someone hands you a trained language model with seventy billion parameters, and asks you to make it cheaper to run. You have a few GPUs for an afternoon, a few hundred example documents, and no budget to retrain anything. What can you do, and what is the best anyone has managed?

That is the question of training-free compression, and it matters more than it sounds. Retraining a large model, even partially, costs real money and needs the original training data, which you usually don't have. Training-free methods take the model as it is, look at how it behaves on a small amount of calibration text, and produce a smaller or cheaper model in minutes to hours, on one or a few GPUs. Almost every compressed model you can download today was made this way.

There are four broad families, and we'll take them in turn. Quantization stores the numbers with fewer bits. Pruning deletes individual weights. Structured compression removes whole rows, columns, dimensions, or matrices, often by replacing a matrix with a low-rank approximation. And depth or expert removal deletes entire layers, or entire experts in a mixture-of-experts model. We'll also look at compressing the attention cache, which is often the real memory bottleneck.

The lecture has a verdict, so let me give it now and then earn it. At equal memory, quantization wins, clearly, and four-bit weight-only quantization is essentially a solved problem. The other families earn their place only in specific situations, or stacked on top of quantization. And the most interesting open problems are statistical ones that the field has mostly ignored.

## What training-free actually means

The phrase hides three different things, and it's worth separating them, because papers blur them.

The strictest tier is data-free. You compress the model using nothing but the weights themselves. Rounding every weight to the nearest four-bit value is data-free. So is a method called HIGGS, which we'll meet shortly, and so is rounding to NVIDIA's four-bit floating-point format.

The middle tier is calibration-only. You run a few hundred sequences of text through the model, record the activations, and use them in a closed-form computation: a least-squares solve, an eigendecomposition, a scaling rule. No gradients, no optimiser. GPTQ, AWQ, SparseGPT, Wanda and SliceGPT all live here.

The loosest tier is calibrated optimisation. The weights themselves are frozen, but small auxiliary parameters, such as a rotation, an affine transform, or a rounding offset, are learned by gradient descent on the calibration data. SpinQuant and FlatQuant are the best-known examples. They don't retrain the model, but they do run an optimiser, and some papers explicitly mark them as not fine-tuning-free.

Keep the three tiers in mind, because comparisons across them are often unfair. A method that runs gradient descent for six hours will usually beat one that solves a closed form in twenty minutes, and the honest question is whether it's worth the difference.

## The objective almost every method shares

Before the families, one idea unites nearly all of them, and it's worth seeing clearly because it's also where the statistics comes in.

Most training-free methods compress one layer at a time. For a linear layer with weights W, they collect the inputs X that layer sees on the calibration data, and they look for a compressed weight matrix, call it W-hat, that makes W-hat times X as close as possible to W times X. That's a least-squares problem: minimise the squared difference between the original layer's outputs and the compressed layer's outputs, over the calibration activations.

Expand that squared error and a single matrix appears: X times its own transpose, the second-moment matrix of the layer's inputs. That matrix is the Hessian of the layer's reconstruction error, and it's the hero of this whole field. It tells you which directions of input space the layer actually uses, and how strongly. A weight that multiplies an input direction with large variance matters a lot. A weight that multiplies a direction that is almost always near zero barely matters at all.

This is the old Optimal Brain Surgeon idea from the early nineties, applied one layer at a time. Remove or round a weight, then use the Hessian to adjust the remaining weights so they compensate for the error. GPTQ does exactly this for quantization: it rounds one column of weights at a time, and after each column it updates all the columns not yet rounded to absorb the rounding error. SparseGPT does the same for pruning. And SliceGPT, which deletes principal directions of the residual stream, is computing an eigendecomposition of essentially the same kind of matrix.

Hold on to that objective, because the most important limitations of the field come from it. It is a layer-by-layer proxy. It ignores how errors in one layer change the inputs to the next. It ignores what the model's actual loss cares about. And it depends entirely on how well a few hundred calibration sequences estimate a matrix with tens of millions of entries.

## Weight-only quantization: four bits is solved

Start with the simplest and most successful family: store each weight in fewer bits, keep the activations in sixteen-bit, and convert the weights back on the fly inside the matrix-multiply kernel.

At four bits per weight, this works. A large evaluation of the Llama three point one models, at eight, seventy and four hundred and five billion parameters, found that four-bit weight-only quantization recovers about ninety-nine point four percent of the original model's accuracy on average. Studies focused on reasoning tasks find drops of about one percent or less. The standard tools, GPTQ and AWQ with a group of a hundred and twenty-eight weights sharing each scale, get you there in minutes to a couple of hours: GPTQ takes about a fifth of a GPU-hour for an eight-billion-parameter model and under two for seventy billion.

Even data-free methods come close. HIGGS uses a random Hadamard rotation to make the weights look Gaussian, then rounds them to a grid designed to be optimal for Gaussian data. It lands within about a third of a perplexity point of GPTQ at four bits, without seeing a single calibration token.

There's one caution worth knowing. Plain GPTQ with one scale per output channel, rather than per group, can fail badly on some newer models. One recent paper reports Llama three seventy billion jumping to a perplexity of about twenty-seven at four bits that way, against under three for the original. Group-wise scales or a rotation fix it. The lesson is that four bits is solved with the right settings, not with any settings.

## Three bits and two bits: where the frontier is

Below four bits, the methods separate.

At three bits, the leaders are QTIP and a newer method called KronQ. QTIP, which you may remember from the trellis lecture, uses a trellis code over rotated weights, so the effective codebook dimension is very large while decoding stays cheap on a GPU. On Llama three eight billion it reaches a perplexity of about six, against five and a half for the original.

At two bits, quality is still the hard problem. QTIP is the clear quality leader: on Llama two seven billion, without any fine-tuning, it reaches a perplexity of about six point eight, against about five point one for the original, and it beats the older vector-quantization methods, QuIP-sharp and AQLM, even when those are allowed to fine-tune. But those vector methods are expensive: published figures put QuIP-sharp at about two hundred and seventy GPU-hours and AQLM at over three hundred for a seventy-billion-parameter model. Cheaper calibration-only methods like KronQ now get Llama three seventy billion to a perplexity of about eight at two bits with only a hundred and twenty-eight calibration samples, though that's a single group's result and still well above the original.

And here's the sobering comparison. One benchmark found that a two-bit seventy-billion model is worse than a four-bit seven-billion model. If your memory budget is fixed, it's often better to quantize a smaller model gently than a larger one brutally. Two bits is a research frontier, not yet a deployment default.

One more cross-cutting finding: newer models are harder to quantize. Models trained on more tokens, like Llama three and Qwen three, and reasoning models in particular, lose more at the same bit width. In one study, Qwen three eight billion at three bits more than doubled its perplexity under AWQ, while Llama three eight billion rose by about a quarter. A plausible reading is that a model trained on more data packs more information into each weight, leaving less slack to round away.

## Quantizing activations too

Weight-only quantization saves memory and speeds up decoding, but the arithmetic still happens in sixteen-bit. To use the low-precision tensor cores, you have to quantize the activations as well, and that's much harder, for one reason: outliers.

Large language models develop a few activation channels with enormous values, sometimes tens of thousands of times larger than typical. A four-bit integer format has only sixteen levels. If one channel is a thousand times larger than the rest, its scale swallows everything, and the ordinary channels all round to zero.

The fixes form a lineage. SmoothQuant divides each activation channel by a factor and multiplies the corresponding weights by the same factor, so the product is unchanged but the difficulty moves from activations to weights. That works well at eight bits. At four bits, the field turned to rotations. QuaRot multiplies the activations by a random Hadamard matrix, which spreads every outlier across all channels, and applies the inverse rotation to the weights, so the network's output is unchanged. This is the same computational-invariance trick SliceGPT uses, and it's training-free. SpinQuant learns the rotation instead of using a random one, by gradient descent on a small calibration set. FlatQuant goes further and learns a general affine transform per layer.

The current results look like this. FlatQuant is the strongest widely replicated result at four-bit weights and four-bit activations. On Llama three eight billion it reaches a perplexity of about seven and a zero-shot average of seventy-one, against about six point one and seventy-three for the original, and its calibration takes about an hour for eight billion and six for seventy. On the same model, QuaRot with plain rounding sits near ten and a half, and SpinQuant between seven and a half and eight. A method called PrefixQuant shows that much of the outlier problem comes from a few specific tokens, and removes it with a training-free step that takes about a minute.

So four-bit activations are close, but not lossless, and the closer you get, the more calibrated optimisation you need. A study of reasoning under quantization recommends FlatQuant for four-bit weights and activations, AWQ for weight-only, and QuaRot for the attention cache.

One more method belongs here because it attacks the shared objective directly. GPTAQ changes a single thing in GPTQ: instead of making the quantized layer match the original layer on the original inputs, it makes the quantized layer, fed its actual, already-quantized inputs, match what the original model would have produced. In other words, each layer corrects the errors of the layers before it. It's closed-form, adds about twenty lines of code, and runs on one GPU. On the four-hundred-billion-parameter Llama, quantized to four-bit weights and activations, it cut perplexity from nearly six to about three and a half compared with GPTQ.

## The hardware formats

The newest GPUs push the choice toward specific formats, and the evidence on them is clearer than the marketing.

Eight-bit floating point, for both weights and activations, is effectively lossless. If your hardware supports it, it's the safe default.

Four-bit floating point comes in two flavours: the industry microscaling format, MX-FP-four, with a shared scale per thirty-two values, and NVIDIA's NV-FP-four, with a finer scale per sixteen values. A careful study published at ICLR this year found they are not equivalent. On a Llama three point one eight-billion instruction model at four-bit weights and activations, NV-FP-four with GPTQ-style rounding recovered about ninety-six percent of the original accuracy. MX-FP-four with plain rounding recovered under eighty-eight, and with a QuaRot rotation, under eighty. The finer scale groups of NV-FP-four absorb much of what rotations were doing, so rotations stop helping and can even hurt.

The authors' summary is worth quoting: four-bit floating point is not an automatic upgrade over four-bit integers. A hypothetical integer format with the same fine scales did slightly better still. With good kernels, these formats give about a two-times end-to-end speedup on the newest data-centre GPUs.

## Compressing the attention cache

For long contexts and big batches, the largest memory cost is often not the weights at all, but the attention cache: the stored keys and values for every previous token. It grows with context length and batch size, and it's read on every decoding step.

The cache compresses well without training. KIVI quantizes keys per channel and values per token to two bits, needs no tuning, and reported over two and a half times less peak memory and roughly two and a half to three and a half times higher throughput. KVQuant reaches under a tenth of a perplexity point of loss at three bits, calibrated on just sixteen samples. And a recent method called TurboQuant, which is data-oblivious, matched the full-precision cache on a long-context benchmark at three and a half bits and lost only a little at two and a half.

There's an important caution from the systems side, though. A 2026 benchmark found that the compression ratio of a cache method doesn't reliably predict its end-to-end speed. Quantizing the cache saves memory, but adds conversion work on every attention step, and at small batch sizes a four-bit cache has been measured to be slower than a sixteen-bit one. The win is real at long context and large batch, which is exactly where you need it.

## Pruning individual weights

Now the second family: setting weights to zero.

The baselines are SparseGPT, which applies the Optimal Brain Surgeon update while pruning, and Wanda, which simply ranks each weight by its magnitude times the norm of its input activation, with no update at all. Wanda's simplicity is the point: one pass, no Hessian inverse, and results close to SparseGPT's at moderate sparsity.

The biggest improvements since then have come not from better scoring but from better allocation. OWL observed that layers with more activation outliers are more sensitive, and gave them lower sparsity, with higher sparsity elsewhere. At seventy percent sparsity on a seven-billion model, that took Wanda's perplexity from about eighty-six down to about twenty-five. Methods like DSnoT, which iteratively swaps pruned and kept weights, and ALPS, which solves the layer problem more exactly, improve on that further.

But two facts keep weight pruning from competing with quantization. The first is quality at equal memory: a recent benchmark on Llama three point one eight billion found four-bit quantization kept essentially all of the model's knowledge-test performance, while fifty-percent pruning kept under ninety percent. And both lost far more on reasoning than on knowledge.

The second is speed. Unstructured sparsity barely speeds anything up on GPUs, because the zeros are scattered. The two-out-of-four pattern, where exactly two of every four consecutive weights are zero, does run on sparse tensor cores, but the measured gains are modest: about one and a half times on individual matrix multiplies and far less end to end, and they shrink at realistic batch sizes. In March of this year, vLLM removed its two-out-of-four kernels as not widely used. Sparsity's best case now is stacking with quantization: a combined four-bit, two-out-of-four kernel reached about three times end to end.

## Narrower and smaller matrices

The third family removes structure: whole dimensions, rows, columns, or replaces matrices with low-rank products, so that every remaining matrix is smaller and dense, and runs on ordinary kernels.

SliceGPT is the clearest example, so let's take a moment on it. It relies on computational invariance: in a transformer with RMS normalisation, you can multiply the residual stream by any orthogonal matrix, and multiply the adjacent weights by its transpose, without changing the network's output. SliceGPT chooses that matrix, layer by layer, from the principal components of the residual stream on calibration data, then deletes the directions with the least variance. Every matrix that touches the residual stream shrinks. The cost is a small extra matrix on each residual connection, to translate between neighbouring layers' bases.

On its own terms it works: slicing a quarter of the residual dimension of the seventy-billion Llama two raised perplexity from about three point three to about four point six, and let the model run on three GPUs instead of four. But next to four-bit quantization, which costs almost nothing in quality for a larger memory saving, that's a weak trade. And its speedups were modest: about eleven to seventeen percent lower latency for a single request, with the real benefit being fewer GPUs.

Its successors fix the quality problem. MoDeGPT decomposes each module jointly with methods matched to its structure, and at thirty percent compression on Llama two seven billion, it reached a perplexity of about seven and a half against SliceGPT's ten and a half, and a zero-shot average about seven points higher. On Llama three eight billion at twenty-five percent, the zero-shot gap was nearly eighteen points.

Low-rank methods replace each weight matrix with a product of two thinner ones. The key improvement was to make the decomposition activation-aware: SVD-LLM whitens the weights by the activation covariance before truncating, so the discarded singular values correspond exactly to output error. Its second version is the training-free leader, reaching a perplexity of about eight on Llama three eight billion at twenty percent compression, against about six for the original, and measured speedups from about one point three times at twenty percent compression to about two point seven at eighty.

There's a systems trap in this whole family. A narrower matrix is only faster if its new dimension suits the hardware. GPU tensor cores work in tiles, and a dimension that isn't a multiple of eight, or better sixty-four, can erase the speedup entirely. One study found a low-rank method at fifteen percent compression with no speedup at all, and a width-pruning method thirty-eight percent slower than the original, purely from awkward dimensions. Rounding to aligned sizes recovered up to one and a half times.

## Removing layers and experts

The coarsest family removes whole layers. Methods like ShortGPT, LaCo and SLEB measure how much each layer changes its input, and delete the least useful ones. This gives the cleanest real speedup of all, close to linear in the fraction removed: about one and a quarter times at twenty percent of layers and about one and a half at thirty-five. But without retraining, quality suffers more than with width methods, especially on generation, and one study found depth pruning particularly harms reasoning that relies on long chains of thought.

Mixture-of-experts models offer a gentler version: remove whole experts. The current leader is a method called REAP, which scores each expert by how strongly the router selects it, times how large its output is. Across models from twenty billion to a trillion parameters, it clearly beat merging similar experts together: on coding tasks, removing a quarter of the experts lost under two percent, against over five percent for merging, and removing half lost about seven percent, against over twenty. One sharp caveat: the calibration data has to match the domain. Calibrating on general web text and then evaluating on code lost essentially everything.

## The verdict at equal memory

So, putting the families side by side at equal memory, three independent benchmark studies agree: four-bit weight-only quantization beats every pruning method. Low-rank and structured methods sit in between, and are worth it mainly when your hardware has no good low-bit kernels, when you want the result to stay a smaller dense model, or when you stack them with quantization.

Two more findings hold across every family. Knowledge-style multiple-choice benchmarks barely move under compression, while reasoning and long generation move a lot, so a compression paper that reports only multiple-choice accuracy is telling you the least important number. And there is no single leaderboard: papers copy each other's baseline numbers, define compression ratios differently, and evaluate at different context lengths. The same SliceGPT setting appears as a perplexity of seven point seven in one paper and eight point two in another. Only comparisons within a single table are safe.

## Where it actually gets faster

A short tour of real speedups, because quality is only half the story.

Four-bit weight-only quantization with good kernels gives close to four times faster matrix multiplies at small batch sizes, falling toward one and a half at a batch of around a hundred and twenty-eight, and up to nearly three times end to end in a serving engine. A large study estimated two to three times lower cost per query at about ninety-nine percent of the accuracy. Four-bit weights and activations, with rotations, give two to three times faster prompt processing and three to four times less peak memory. Eight-bit floating point is close to free in quality and is supported everywhere. Two-out-of-four sparsity gives about one and a half times at best. Depth pruning gives near-linear speedups. Width and low-rank methods give modest speedups, if and only if the dimensions are aligned.

On deployability, the major serving engines all support the main weight-only formats, eight-bit floating point, and four-bit floating point on new hardware. None of them supports a sliced model out of the box; you'd need a custom model class. And here's a gap worth noticing: I found no benchmark that runs every family through one serving engine on one GPU. That benchmark would settle a lot of arguments.

## Improving it statistically

Now the part I find most interesting, because it's where a statistician has the most to add. Training-free compression is, at heart, an estimation problem: we estimate a few large matrices from a small, dependent sample, then make a decision based on them. The field mostly treats those estimates as exact. Here are the places where better statistics should help. Some have published results; others are my own proposals, and I'll say which.

The first problem is the objective. Each layer is compressed to match its own outputs, not to preserve the model's loss. SliceGPT's version of this is especially clear: principal component analysis keeps the directions with the most variance, but the error that matters is variance weighted by how much the next layers amplify it. Plain PCA is optimal only if the next layer treats all directions equally, which it doesn't. Whitening-based methods like SVD-LLM fix this for low-rank compression: whitening by the activation covariance makes truncation exactly equivalent to minimising output error, which is a reduced-rank regression. Fisher-weighted decompositions go a step further toward the actual loss. And GPTAQ, which we met earlier, addresses error propagation between layers by calibrating each layer on the already-compressed inputs it will really receive. A statistician would recognise that as an errors-in-variables regression: the predictors themselves are noisy, and ignoring it biases the fit.

The second problem is how much data we really have. A typical calibration set is about a thousand sequences of two thousand tokens, and the layer dimension is four to eight thousand. If the tokens were independent, two million samples for a four-thousand-dimensional covariance would be plenty. But tokens within a sequence are strongly correlated, so the effective sample size lies somewhere between the number of sequences and the number of tokens, and it's rarely measured. My own back-of-envelope: if the effective sample size is closer to the sequence count, the ratio of dimension to samples is about four, which is deep in the regime where random matrix theory says sample eigenvalues are biased, and small eigen-directions are essentially noise. If it's closer to the token count, the estimates are fine. That single unknown decides whether the bottom of the spectrum, exactly the part slicing and low-rank methods throw away, is being estimated or invented.

Third, shrinkage. GPTQ already adds a small multiple of the identity to its Hessian before inverting it, which is a crude shrinkage estimator. A principled version, such as Ledoit-Wolf shrinkage, would choose that amount from the data. But note a subtlety, again my own reasoning: shrinking toward a multiple of the identity changes eigenvalues but not eigenvectors, so it helps methods that invert the matrix, like GPTQ and whitening, but does nothing for methods that only use its principal directions, like SliceGPT. Those need a different target or a robust estimator.

Fourth, heavy tails. Activations contain a few enormous outliers, which dominate any sample covariance. Robust estimators exist, but the classic ones are too expensive at this dimension. Cheaper options, which I'd propose rather than cite, include estimating the covariance on token-normalised activations, which matches what RMS normalisation does anyway, and keeping the outlier channels in a small, always-retained subspace, rather than letting them distort the estimate of everything else. Notably, the outliers shouldn't simply be clipped, because they carry real function in the model.

Fifth, calibration design. The choice of calibration text matters more than most papers admit. Several studies show domain effects, and one 2026 study found that simply redrawing the calibration sample reversed the ranking of GPTQ and AWQ. Expert pruning on code with web-text calibration failed completely. Practical recipes include mixing calibration domains deliberately, or choosing weights to protect the worst-off domain, a maximin version of the objective.

Sixth, uncertainty. Every compressed model is a decision made from a random sample, yet no method reports how stable that decision is. A cheap proposal: compute the calibration statistics per group of sequences, then resample those groups, by jackknife or bootstrap, and report a confidence interval on the quantities that matter, like the variance retained or the error at each layer's chosen size. The cut point itself can be chosen with tools statisticians already have, such as parallel analysis or the Gavish-Donoho optimal threshold for singular values. To my knowledge no published work applies these formally to SliceGPT-style compression, so this is genuinely open.

And seventh, allocation. The biggest single improvement in weight pruning came from giving different layers different budgets, which is OWL's contribution. The general version is a constrained optimisation: estimate each layer's error as a function of how much you compress it, then choose the per-layer budgets to minimise total error under a size constraint, a knapsack or Lagrangian problem. Using bootstrap-stabilised sensitivity estimates there, rather than single noisy ones, is another of my proposals.

## Improving it for efficiency

On the systems side, the improvements are more concrete.

Align every compressed dimension to the hardware's tile sizes; it's free and can be the difference between a speedup and a slowdown. Prefer formats the hardware runs natively, eight-bit floating point everywhere and four-bit floating point with fine-grained scales on the newest GPUs, over formats that need software decoding. Stack compatible methods: quantization with structured compression, or quantization with cache compression, each targeting a different cost. Make the compression procedure itself cheaper: SliceGPT spends about three and a half hours on one GPU for seventy billion parameters, mostly on eigendecompositions in double precision, which randomized or streaming methods could cut. And build serving support for the structural methods; until a sliced or low-rank model loads in a standard engine, it remains a research artefact.

## What to remember

Let me close with the key ideas.

Training-free compression means no retraining of weights, but it spans three tiers, from data-free rounding to calibrated optimisation of rotations, and fair comparisons stay within a tier. Nearly all methods share one objective: make each compressed layer reproduce its original outputs on calibration data, with the input second-moment matrix as the Hessian. Four-bit weight-only quantization is solved and is the default. Two bits is a frontier, led by trellis codes, and a gently quantized small model often beats a brutally quantized large one. Four-bit activations work with rotations or learned transforms but aren't lossless. Eight-bit floating point is free, and four-bit floating point needs fine-grained scales. The attention cache compresses well at three to four bits. Weight pruning, structured slicing, low-rank, and depth removal all lose to quantization at equal memory, and are best used where quantization can't go, or stacked on top of it. And the statistics underneath, effective sample size, heavy tails, calibration design, uncertainty and allocation, are mostly unaddressed, which makes them the most promising place to improve.

## Self-check questions

Eight questions, with brief answers.

One. What are the three tiers of training-free compression? Data-free, calibration-only with closed-form computation, and calibrated optimisation of auxiliary parameters such as rotations, with the weights frozen.

Two. What objective do most methods share, and what matrix sits at its heart? Matching each compressed layer's outputs to the original on calibration data; the second-moment matrix of the layer's inputs, which acts as the Hessian.

Three. Why are activations harder to quantize than weights, and how do rotations help? A few channels carry huge outliers that swamp a four-bit scale; an orthogonal rotation spreads them across all channels, while computational invariance keeps the network's output unchanged.

Four. Why can a two-bit seventy-billion model lose to a four-bit seven-billion one? At two bits the quantization error is large enough to outweigh the bigger model's advantage; at a fixed memory budget, gentle compression of a smaller model can win.

Five. Why don't rotations help NVIDIA's four-bit floating-point format much? Its scale is shared by only sixteen values, so the fine-grained scaling already absorbs most of the outlier problem that rotations were solving.

Six. What does GPTAQ change about GPTQ? It calibrates each layer on the inputs it will actually receive from the already-quantized earlier layers, so it corrects propagated errors, at almost no extra cost.

Seven. Why might the bottom of an activation spectrum be unreliable? Tokens within a sequence are correlated, so the effective sample size may be close to the number of sequences, putting the estimate in the regime where small eigenvalues and their directions are mostly noise.

Eight. Why can a narrower matrix fail to run faster? If its new dimension isn't aligned to the GPU's tile sizes, the kernel wastes work or falls back to slower paths, and the speedup can vanish or reverse.

That's the state of training-free compression. Thanks for listening.
