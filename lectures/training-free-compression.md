# Compression Without Retraining

### The state of the art in training-free model compression, and where it goes next

*About forty minutes spoken. A survey lecture, built from a literature review of papers up to late twenty twenty-six, and updated in October with a uniform head-to-head study of nineteen of these methods under one protocol. Numbers are rounded; exact tables are in the review notes. Where a claim is my own reasoning rather than a published result, I'll say so.*

---

## Why training-free

Suppose someone hands you a trained language model with seventy billion parameters, and asks you to make it cheaper to run. You have a few GPUs for an afternoon, a few hundred example documents, and no budget to retrain anything. What can you do, and what is the best anyone has managed?

That is the question of training-free compression. Training-free methods take the model as it is, look at how it behaves on a small amount of calibration text, and produce a smaller or cheaper model in minutes to hours.

There are four broad families. Quantization stores the numbers with fewer bits. Pruning deletes individual weights. Structured compression removes whole rows, columns or dimensions, or replaces a matrix with a low-rank approximation. And depth or expert removal deletes entire layers, or entire experts in a mixture-of-experts model.

The lecture has a verdict, so let me give it now and then earn it. At equal memory, quantization wins, clearly, and four-bit weight-only quantization is essentially a solved problem. The other families earn their place only in specific situations, or stacked on top of quantization. And the most interesting open problems are statistical ones that the field has mostly ignored.

Later on we'll meet a uniform study that put nineteen training-free methods through one identical protocol. Most of the verdict survives it, and the study sharpens it in several places.

## What training-free actually means

The phrase hides three different things, and papers blur them.

The strictest tier is data-free: you compress using nothing but the weights themselves. Rounding every weight to the nearest four-bit value is data-free. So is a method called HIGGS, which we'll meet shortly, and so is rounding to NVIDIA's four-bit floating-point format.

The middle tier is calibration-only. You run a few hundred sequences of text through the model, record the activations, and use them in a closed-form computation: a least-squares solve, an eigendecomposition, a scaling rule. No gradients, no optimiser. GPTQ, AWQ, SparseGPT, Wanda and SliceGPT all live here.

The loosest tier is calibrated optimisation. The weights are frozen, but small auxiliary parameters, such as a rotation, an affine transform or a rounding offset, are learned by gradient descent on the calibration data. SpinQuant and FlatQuant are the best-known examples. They don't retrain the model, but they do run an optimiser, and some papers explicitly mark them as not fine-tuning-free.

Keep the tiers in mind, because comparisons across them are often unfair. A method that runs gradient descent for six hours will usually beat one that solves a closed form in twenty minutes, and the honest question is whether it's worth the difference.

## The objective almost every method shares

Before the families, one idea unites nearly all of them, and it's also where the statistics comes in.

Most training-free methods compress one layer at a time. For a linear layer with weights W, they collect the inputs X that layer sees on the calibration data, and look for a compressed matrix, call it W-hat, that makes W-hat times X as close as possible to W times X. That's a least-squares problem over the calibration activations.

Expand that squared error and a single matrix appears: X times its own transpose, the second-moment matrix of the layer's inputs. Up to a factor of two, that is the Hessian of the layer's reconstruction error, and the same matrix serves every row of weights. It's the hero of this whole field: it tells you which input directions the layer actually sees, and how strongly.

This is the old Optimal Brain Surgeon idea from the early nineties, applied one layer at a time. Remove or round a weight, then use the inverse of the Hessian to adjust the remaining weights so they compensate. GPTQ does exactly this for quantization: it rounds one column at a time, and after each column it updates the columns not yet rounded to absorb the error. SparseGPT does the same for pruning. And SliceGPT, which deletes principal directions of the residual stream, is computing an eigendecomposition of essentially the same matrix.

Hold on to that objective, because the field's most important limitations come from it. It is a layer-by-layer proxy. It ignores how errors in one layer change the inputs to the next. It ignores what the model's actual loss cares about. And it depends entirely on how well a calibration set of a hundred or so sequences, a few hundred thousand tokens in all, estimates a matrix with tens of millions of entries.

## Weight-only quantization: four bits is solved

Start with the simplest and most successful family: store each weight in fewer bits, keep the activations in sixteen-bit, and convert the weights back on the fly inside the matrix-multiply kernel.

At four bits per weight, this works. The reference is a large evaluation of the Llama three point one family, from eight billion up to four hundred and five billion parameters. It found that four-bit weight-only quantization recovers about ninety-nine point four percent of the original accuracy on average across the standard academic benchmarks. On a harder suite, it recovered about ninety-six percent for the eight-billion model and ninety-seven for the seventy-billion. Studies focused on reasoning tasks find drops of about one percent or less. The standard tools are GPTQ and AWQ, with a group of a hundred and twenty-eight weights sharing each scale, and they run in minutes to a couple of hours.

Even data-free methods come close. HIGGS uses a random Hadamard rotation to make the weights look Gaussian, then rounds them to a grid designed for Gaussian data. At about four bits it matches or slightly beats GPTQ on perplexity in its own paper's table, without seeing a single calibration token.

One caution. Plain GPTQ with one scale per output channel, rather than per group, can fail badly on some newer models. One recent paper reports the seventy-billion Llama three model jumping to a perplexity of about twenty-seven at four bits that way, against under three for the original. Group-wise scales or a rotation fix it.

## Three bits and two bits: where the frontier is

Below four bits, the methods separate.

At three bits, the leaders are QTIP and a newer method called KronQ. QTIP, which you may remember from the trellis lecture, uses a trellis code over rotated weights. On the eight-billion Llama three model it reaches a perplexity of about six, against five and a half for the original.

At two bits, quality is still the hard problem. QTIP leads among methods that don't fine-tune: on the seven-billion Llama two model it reaches a perplexity of about six point eight, against about five point one for the original. That is competitive with the older vector-quantization methods, QuIP-sharp and AQLM, even when those are allowed to fine-tune. Those methods are expensive: published figures put QuIP-sharp at about two hundred and seventy GPU-hours and AQLM at over three hundred for a seventy-billion-parameter model. KronQ needs just one backward pass over the calibration set and only a hundred and twenty-eight calibration samples. It gets the seventy-billion Llama three model to a perplexity of about eight at two bits, against under three for the original, though that's a single group's result.

A sobering comparison. One benchmark found that, with the older two-bit method it tested, a seventy-billion model did worse than a seven-billion model at four bits. And the two-bit model takes roughly five times the memory. Better two-bit methods like QTIP do much better at seventy billion, so two-bit results depend heavily on the method. But the direction is worth remembering: a gently quantized smaller model can beat a brutally quantized larger one.

One more cross-cutting finding: newer models are harder to quantize. Models trained on more tokens, like Llama three and Qwen three, and reasoning models in particular, lose more at the same bit width. In one study at three bits under AWQ, the eight-billion-parameter Qwen three model more than doubled its perplexity. The eight-billion Llama three model rose by about a quarter.

## Quantizing activations too

Weight-only quantization saves memory and speeds up decoding, but the arithmetic still happens in sixteen-bit. To use the low-precision tensor cores, you have to quantize the activations as well, and that's much harder, for one reason: outliers.

Large language models develop a few activation channels with enormous values, sometimes tens of thousands of times larger than typical. A four-bit integer format has only sixteen levels. If one channel is a thousand times larger than the rest and they all share one scale, that scale is set by the outlier, and the ordinary channels all round to zero.

SmoothQuant divides each activation channel by a factor and multiplies the corresponding weights by the same factor, so the product is unchanged but the difficulty moves to the weights. That works well at eight bits. At four bits, the field turned to rotations. QuaRot, which is training-free, multiplies the activations by a random Hadamard matrix, which spreads every outlier across all channels, and applies the inverse rotation to the weights, so the network's output is unchanged. Why this helps is simple geometry: a rotation preserves length, so one huge channel becomes many moderate ones, each shrunk by about the square root of the number of channels, and rounding error depends on the ratio of the largest entry to a typical one, not on the length. SpinQuant learns the rotation by gradient descent on a small calibration set instead of drawing it at random. FlatQuant goes further and learns a general affine transform per layer.

The current results: FlatQuant and PrefixQuant are among the strongest widely replicated integer results at four-bit weights and four-bit activations. On the eight-billion Llama three model, FlatQuant reaches a perplexity of about seven, against six point one for the original. Its zero-shot average is seventy-one, against seventy-three. On the same model, QuaRot with plain rounding sits near ten and a half, and SpinQuant between seven and a half and eight. PrefixQuant instead pins the few specific tokens that cause much of the outlier problem, a training-free step that takes about a minute.

So four-bit activations are close, but not lossless, and the closer you get, the more calibrated optimisation you need.

One more method attacks the shared objective directly. GPTAQ changes a single thing in GPTQ: the quantized layer, fed its actual, already-quantized inputs, is made to match what the original model would have produced, so each layer corrects the errors of the layers before it. It's closed-form and cheap. On the four-hundred-and-five-billion Llama at four-bit weights and activations, GPTQ gave a perplexity of nearly six. GPTAQ cut that to about three and a half, against about one and a half for the original.

## The hardware formats

The newest GPUs push the choice toward specific formats.

Eight-bit floating point, for both weights and activations, is effectively lossless on the GPUs that support it, and is the safe default there.

Four-bit floating point comes in two flavours: the industry microscaling format, MX-FP-four, with a shared scale per thirty-two values, and NVIDIA's NV-FP-four, with a finer scale per sixteen values. A careful study published at I-C-L-R this year found they are not equivalent. The test model was the eight-billion-parameter instruction model from Llama three point one, at four-bit weights and activations. NV-FP-four with GPTQ-style rounding recovered about ninety-six percent of the original accuracy. MX-FP-four with plain rounding recovered under eighty-eight, and with a QuaRot rotation, under eighty. The authors' reading, which I find plausible, is that the finer scale groups of NV-FP-four absorb much of what rotations were doing: an outlier then only spoils the sixteen values that share its scale, and a rotation that smears it into every group can make things worse for the rest. One caution: NV-FP-four also uses a finer, non-power-of-two scale, so the comparison doesn't isolate group size alone.

Their summary: four-bit floating point is not an automatic upgrade over four-bit integers. With good kernels, these formats give about a two-times end-to-end speedup on the newest data-centre GPUs.

## Compressing the attention cache

For long contexts and big batches, the largest memory cost is often not the weights but the attention cache: the stored keys and values for every previous token, read on every decoding step.

The cache compresses well without training. KIVI quantizes keys per channel and values per token to two bits, with no tuning. It reported over two and a half times less peak memory and roughly two and a half to three and a half times higher throughput. KVQuant reaches under a tenth of a perplexity point of loss at three bits, calibrated on just sixteen samples.

The systems caution: a twenty twenty-six benchmark found that a cache method's compression ratio doesn't reliably predict its end-to-end speed. Quantizing the cache saves memory but adds conversion work on every attention step, and at small batch sizes a four-bit cache has been measured to be slower than a sixteen-bit one. The win is real at long context and large batch, which is exactly where you need it.

## Pruning individual weights

Now the second family: setting weights to zero.

The baselines are SparseGPT, which applies the Optimal Brain Surgeon update while pruning, and Wanda, which simply ranks each weight by its magnitude times the norm of its input activation, with no update at all, and comes close to SparseGPT at moderate sparsity.

Some of the biggest improvements since then have come from allocation rather than scoring. OWL observed that layers with more activation outliers are more sensitive, and gave them lower sparsity, with higher sparsity elsewhere. At seventy percent sparsity on a seven-billion model, Wanda's perplexity was about eighty-six. OWL took it down to about twenty-five.

But two facts keep weight pruning from competing with quantization. The first is quality at equal memory. In a recent benchmark on the eight-billion model from Llama three point one, four-bit quantization kept essentially all of the model's knowledge-test performance, while fifty-percent pruning kept under ninety percent. Both lost more on reasoning than on knowledge, though how much a four-bit model loses on reasoning varies a lot from study to study.

The second is speed. Unstructured sparsity barely speeds anything up on GPUs, because the zeros are scattered. The two-out-of-four pattern, exactly two zeros in every four consecutive weights, does run on sparse tensor cores, but the gains are modest: about one and a half times on individual matrix multiplies, far less end to end, and shrinking at realistic batch sizes. In March of this year, V-L-L-M removed its two-out-of-four kernels as not widely used. Sparsity's best case now is stacking with quantization: a combined four-bit, two-out-of-four kernel reached up to about three times end to end.

## Narrower and smaller matrices

The third family removes structure: whole dimensions, rows or columns, or replaces matrices with low-rank products, so that every remaining matrix is smaller and dense, and runs on ordinary kernels.

SliceGPT is the clearest example. It relies on computational invariance: in a transformer with RMS normalisation, once the norm's learned scale has been folded into the neighbouring weights, you can multiply the residual stream by any orthogonal matrix, and the adjacent weights by its transpose, without changing the network's output, because an orthogonal matrix leaves a vector's length unchanged and the normalisation can't tell. SliceGPT chooses that matrix, layer by layer, from the principal components of the residual stream on calibration data, then deletes the directions with the least variance. The cost is an extra square matrix on each residual connection, as wide as the sliced residual stream, to translate between neighbouring layers' bases.

On its own terms it works. Slicing a quarter of the residual dimension of the seventy-billion Llama two raised perplexity from about three point three to about four point six. It also let the model run on three GPUs instead of four. But next to four-bit quantization, which costs almost nothing in quality for a larger memory saving, that's a weak trade. And its speedups were modest: eleven to seventeen percent lower latency for a single request, with the real benefit being fewer GPUs.

Successors such as MoDeGPT report much better quality at the same ratio in their own tables. At thirty percent compression on the seven-billion Llama two model, MoDeGPT reached a perplexity of about seven and a half. SliceGPT's was ten and a half.

Low-rank methods replace each weight matrix with a product of two thinner ones. The key improvement was to make the decomposition activation-aware: SVD-LLM whitens the weights by the activation covariance before truncating, so the discarded singular values correspond exactly to output error. Its second version is reported as the training-free leader in its own tables. On the eight-billion Llama three model at twenty percent compression, it reaches a perplexity of about eight. The original is about six. Hold that ranking loosely; the uniform study later in this lecture puts a different low-rank method first.

There's a systems trap in this whole family. A narrower matrix is only faster if its new dimension suits the hardware. A dimension that isn't a multiple of eight, or better sixty-four, can erase the speedup entirely. One study found a low-rank method at fifteen percent compression with no speedup at all, and a width-pruning method thirty-eight percent slower than the original, purely from awkward dimensions. Rounding to aligned sizes recovered up to one and a half times.

## Removing layers and experts

The coarsest family removes whole layers. Methods like ShortGPT, LaCo and SLEB delete or merge the layers that matter least. This gives the cleanest real speedup of all, close to linear in the fraction removed. At batch size one, removing a fifth of the layers gave about one and a quarter times faster decoding. Removing a third gave about one and a half. Without retraining, quality tends to suffer more than with width methods, especially on generation, although the comparisons are mixed, and one study found depth pruning particularly harms reasoning that relies on long chains of thought.

Mixture-of-experts models offer a gentler version: remove whole experts. The current leader is REAP, which scores each expert by how strongly the router selects it, times how large its output is. It clearly beat merging similar experts on generative tasks such as coding. Removing a quarter of the experts lost under two percent on coding, against over five percent for merging. Removing half lost about seven percent, against over twenty. One sharp caveat: the calibration data has to match the domain. Calibrating on general web text and then evaluating on code lost essentially everything. And a second caveat: deleting experts shrinks the model on disk, but not necessarily what each token reads. The uniform study later makes that concrete, and on perplexity it also ranks pruning and merging the other way round.

## The verdict at equal memory

Putting the families side by side at equal memory, three independent benchmark studies agree: four-bit weight-only quantization beats every pruning method. Low-rank and structured methods sit in between, and are worth it mainly when your hardware has no good low-bit kernels, when you want the result to stay a smaller dense model, or when you stack them with quantization.

Two more findings hold across every family. Knowledge-style multiple-choice benchmarks barely move under compression, while reasoning and long generation move a lot, so a paper that reports only multiple-choice accuracy is telling you the least important number. And there is no single leaderboard: papers copy each other's baseline numbers, define compression ratios differently, and evaluate at different context lengths. The same SliceGPT setting appears as a perplexity of seven point seven in one paper and eight point two in another. Only comparisons within a single table are safe, which is why the next chapter's study was worth running.

## A uniform head-to-head study

The study took two models from the Qwen three family. One is a small dense model with one point seven billion parameters. The other is a mixture-of-experts model with thirty billion parameters, of which about three billion are active for each token. It covered nineteen training-free methods across the two models, with one calibration set and one evaluation protocol for every point. Perplexity was measured on WikiText and on the C-four web corpus. Each method was asked to remove a fifth, about a third, or half of the bytes in the model's linear layers. Every point was measured twice: on its own in sixteen-bit, and stacked with four-bit microscaling, the MX-FP-four format we met earlier. The whole thing came to about a hundred and sixty verified results, in about three and a half hours of rented GPU time, for under twenty dollars. A few things weren't run: one method at the two larger budgets and a search method called DarwinLM, for lack of time, and two mixture-of-experts variants.

The results were ranked on decode bytes per token: how many bytes of weights the hardware must read to generate each new token. At small batch, decoding is limited by memory traffic, so time tracks bytes read; as the batch grows, the hardware becomes compute-limited and byte savings stop translating into speed. The measure also leaves out the attention cache, which at long context can dominate. Stored bytes, the size of the model on disk, were reported alongside, and the gap between the two is one of the study's main lessons.

The starting points: the dense model scores a C-four perplexity of about twenty-two point four, rising to about twenty-six with plain four-bit microscaling. The mixture-of-experts model scores about fourteen, rising to about fifteen and a half.

Now the seven findings.

First, unstructured sparsity is the strongest of the removal families, as measured by perplexity. SparseGPT, with Wanda close behind, removed a fifth of the dense model's linear bytes for essentially no change in perplexity. A third cost around six percent. Half cost about thirty. On the mixture-of-experts model the costs were smaller still: about three percent at a third. At half, about ten. But there's a catch. Those savings only become real bytes if the hardware stores and executes a sparse format; on hardware that stores zeros densely, you've saved nothing. And a sparse format has to record where the survivors are. Take a simple one-bit mask at half sparsity. In sixteen-bit, that's nine bits per original weight against sixteen. In four-bit, it's three against four. So a halving becomes a cut of about a quarter, before counting the scales. And the structured patterns that hardware does support, two out of four and four out of eight, were clearly worse: at half sparsity, two-out-of-four SparseGPT nearly doubled the perplexity.

Second, and this is the finding I'd underline: on the dense model, no structured, low-rank or weight-sharing method beat simply using a smaller model from the same family. The six-hundred-million-parameter Qwen three model is about three times smaller than the one point seven billion model. It scores a perplexity of about thirty. The best structured methods, removing only a fifth of the bytes, landed at about thirty to thirty-six. So they paid the same quality, or worse, for far less compression. Beyond a fifth, almost every structured or low-rank method more than tripled the original perplexity. If a smaller sibling of your model exists, that's the baseline any structured compression has to beat, and at this scale none did.

Third, on the mixture-of-experts model, the cheapest saving in decode bytes was simply activating fewer experts per token. The model normally routes each token to eight experts. Cutting that to six, in a variant the study labels L-D-A, cost under one percent of perplexity for about a seventh fewer decode bytes. Cutting to five cost under two and a half percent for a little over a fifth fewer. Stacked with four-bit microscaling, five experts per token cut decode bytes to just over a fifth of the original. The perplexity was about sixteen, against fifteen and a half for microscaling alone and fourteen for the original. In sixteen-bit, unstructured sparsity was slightly better than cutting experts, but it lost that advantage under four-bit microscaling, because the mask metadata eats a large share of the saving, as we just saw. Once quantization was applied, fewer experts per token was clearly the best option on this model.

Fourth, removing or merging experts shrinks storage, not decode bytes. If the router still picks eight experts for every token, the bytes read per token don't change, however many you delete. In this study, REAP cost about six percent in perplexity with a fifth of the experts removed. With half removed, it cost nearly half. Merging experts was actually better at a fifth, costing about one and a half percent, and then collapsed beyond it. That ordering differs from the REAP paper's. My reading, not the study's, is that they measure different things: the paper measured generative coding tasks, and found merging does fine on multiple-choice tests, while the study measured perplexity. Pruning the width inside each expert does cut decode bytes, but cost about fifteen percent at a fifth.

Fifth, low rank was weak, and the ranking differs from the literature's. The best low-rank method was one called Zero-Sum SVD, and even it lost about half again in perplexity at a fifth of the bytes removed. Then came a cluster of near-equals, including SVD-LLM and its second version. Interestingly, that second version's choice of a different rank for each layer did worse than plain uniform ranks here. And on the mixture-of-experts model, applying low rank to the attention layers alone was catastrophic.

Sixth, a free improvement. GPTQ combined with a search over the shared exponent scales costs no extra bytes compared with plain rounding to four-bit microscaling. It closed more than half of the quality gap on the dense model, from about twenty-six back to about twenty-four. On the mixture-of-experts model it closed more than a quarter. It cost nothing here, so it's worth trying if you ship microscaling formats.

And seventh, stacking can fail outright. SliceGPT collapsed when combined with four-bit microscaling, its perplexity going from the tens and low hundreds to the hundreds of thousands. The most plausible reason is those extra shortcut matrices on the residual connections. They're dense, they act on the residual stream, which carries the largest outliers, and they sit in every layer, so their rounding errors compound with depth. The study attributes the collapse to them, but as far as I know it didn't isolate that by keeping just those matrices in sixteen-bit. And three attention-compression methods that are normally followed by recovery training, namely Palu, M-H-A-to-M-L-A and DeltaLLM, were unusable without it.

One caveat, from the study's author: a one point seven billion model is pessimistic for structural methods. In a separate measurement on the two hundred and thirty-five billion Qwen model, the SliceGPT penalty was about a third of the small model's. My guess is that small models have less redundancy to remove. So the numbers will move with scale, and whether the family ordering holds is a hypothesis rather than a result: that check was one method at one point, on a model that is itself a mixture of experts, so scale and architecture are mixed together.

A word on what uniform does and doesn't buy. It removes differences in calibration data, evaluation text and byte accounting, and because every method is scored on the same evaluation text, the differences are paired, so evaluation noise is small. What it doesn't remove is noise from the calibration draw. Each point is a single run with one calibration sample and no confidence interval, and as we'll see in the statistics chapter, redrawing the calibration sample has been reported to reverse the ranking of two methods. So a doubling or tripling of perplexity is safe to read, but small gaps, like two percent against one and a half, or near-ties among the low-rank methods, shouldn't be read as orderings. And this is perplexity only; compression hurts reasoning and long generation more than simple tests show, so these costs are likely lower bounds on what you'd see downstream.

Put together, the study sharpens the verdict rather than overturning it, within its limits: two models, one scale each, perplexity only. Four-bit quantization done well, meaning GPTQ with the exponent search, gave nearly four times compression for about seven percent of perplexity on the dense model. Plain rounding cost about seventeen percent. None of the removal methods measured on their own came close to that trade. Among them, measure what your hardware actually reads at your batch size, not what the file stores; on a dense model like this one, beat a smaller sibling or don't bother; and on this fine-grained mixture-of-experts model, the cheapest lever was how many experts each token uses, not how many exist.

## Where it actually gets faster

Four-bit weight-only quantization with good kernels gives close to four times faster matrix multiplies at small batch sizes. That falls toward one and a half times at a batch of around a hundred and twenty-eight: the same roofline, since the saving is in bytes read. End to end in a serving engine, it gives up to nearly three times, and a large study estimated two to three times lower cost per query at about ninety-nine percent of the accuracy. Four-bit weights and activations, with rotations, gave two to three times faster prompt processing and three to four times less peak memory, in the QuaRot paper's research kernels on a consumer GPU.

On deployability, the major serving engines all support the main weight-only formats, eight-bit floating point, and four-bit floating point on new hardware. I found no sign that any of them supports a sliced model out of the box; you'd likely need a custom model class. I found no benchmark that runs every family through one serving engine on one GPU. The uniform study counted bytes; a timed version of it on one engine would settle a lot of arguments.

## Improving it statistically

Now the part I find most interesting, because it's where a statistician has the most to add. Training-free compression is, at heart, an estimation problem: we estimate a few large matrices from a small, dependent sample, then make a decision based on them. The field mostly treats those estimates as exact. Here are seven places where better statistics should help. Some have published results; others are my own proposals, and I'll say which.

First, the objective. Each layer is compressed to match its own outputs, not to preserve the model's loss. SliceGPT's version is especially clear: principal component analysis keeps the directions with the most variance, but the error that matters is variance weighted by how much the next layers amplify it, and plain PCA is optimal only if the next layer treats all directions equally, which it doesn't. Whitening by the activation covariance, as SVD-LLM does, fixes this for low-rank compression: truncation then minimises output error on the calibration data exactly, a reduced-rank regression, but it needs an invertible, well-conditioned covariance, which is exactly where the sample-size problem below bites. And GPTAQ, which we met earlier, addresses error propagation by calibrating each layer on the already-compressed inputs it will really receive. A statistician would recognise a cousin of errors-in-variables regression, though not the textbook case, since the noise is deterministic and correlated with the signal. The cleaner statement is: fit on the input distribution you'll actually deploy on.

Second, how much data we really have. A common default is a hundred and twenty-eight sequences of about two thousand tokens, around a quarter of a million tokens in all. The layer dimension is four to eight thousand. Tokens within a sequence are strongly correlated, so the effective sample size lies somewhere between the number of sequences and the number of tokens. My own back-of-envelope: if the effective sample size were as small as the sequence count, the ratio of dimension to samples would be about thirty, far past one. In that regime the sample covariance is rank-deficient, random matrix theory says its eigenvalues are spread out and biased, and the small eigen-directions are essentially noise. Close to the token count, the ratio would be about one in sixty and the estimates would be fine. The truth is in between, and it isn't one number: slowly varying directions, like topic, have few effective samples, while token-level directions have many. Heavy tails make it worse. So effective sample size, how fast the spectrum decays, and tail weight together decide whether the bottom of the spectrum, exactly the part slicing and low-rank methods throw away, is being estimated or invented.

The sharpest version of this is in mixture-of-experts models. Each expert's statistics come only from the tokens routed to it. With a hundred and twenty-eight experts and eight chosen per token, the average expert sees about a sixteenth of the calibration tokens, and rare experts see far fewer. So an expert's covariance, or a pruning score, may rest on very little data, and the rarely used experts are exactly the ones pruning wants to delete. That's my reasoning, not a published result, but it suggests reporting tokens per expert, and remembering that unused on the calibration set isn't the same as unused.

Third, shrinkage. GPTQ already adds a small multiple of the identity to its Hessian before inverting it, a crude shrinkage estimator, though its purpose is mainly numerical stability. A principled version, such as Ledoit-Wolf shrinkage, would choose that amount from the data, though it optimises covariance accuracy rather than the compression objective, so the best amount for downstream error may differ. But note a subtlety, again my own reasoning: shrinking toward a multiple of the identity changes eigenvalues but not eigenvectors. So it helps methods that invert the matrix, like GPTQ and whitening, but it can't change which principal subspace a method like SliceGPT keeps, only eigenvalue-based decisions such as how many components to keep. Better eigenvectors need more data, structure such as a factor model, or a robust estimator.

Fourth, heavy tails. Activations contain a few enormous outliers, which dominate any sample covariance, and the classic robust estimators are too expensive at this dimension. Two cheap options, which I'd propose rather than cite. One is to estimate the covariance on token-normalised activations. That already happens for layers that read the output of an RMS normalisation, so it matters where the input isn't normalised, such as the feed-forward down projection and the raw residual stream that SliceGPT works on. The other is to keep the outlier channels in a small, always-retained subspace, rather than letting them distort the estimate of everything else. The outliers shouldn't simply be clipped, because they carry real function in the model.

Fifth, calibration design. The choice of calibration text matters more than most papers admit. Several studies show domain effects, and one twenty twenty-six study reported that, in most of its test cases, simply redrawing the calibration sample was enough to reverse the ordering of GPTQ and AWQ. Practical recipes include mixing calibration domains deliberately, or choosing weights to protect the worst-off domain, a maximin version of the objective.

Sixth, uncertainty. Every compressed model is a decision made from a random sample, yet no method reports how stable that decision is. A cheap proposal: compute the calibration statistics per group of sequences, and resample whole groups, since sequences, not tokens, are the roughly independent units. Use a jackknife or bootstrap to put an interval on smooth quantities, like the error at each layer's chosen size. Two cautions. Variance retained should be measured on held-out sequences, with the subspace fitted on the rest, because in-sample it's optimistic. And in this high-dimensional regime the bootstrap is unreliable for the eigenvalues and eigenvectors themselves. Familiar cut-point tools, parallel analysis and the Gavish-Donoho threshold, assume independent noise with a clean signal-plus-noise structure that activations don't have, so treat them as diagnostics, not answers. To my knowledge nobody has applied these formally to SliceGPT-style compression, so this is genuinely open.

And seventh, allocation. One of the biggest improvements in weight pruning came from giving different layers different budgets, OWL's contribution. The general version is a constrained optimisation: estimate each layer's error as a function of how much you compress it, then choose per-layer budgets to minimise total error under a size constraint. That assumes errors add across layers, which propagation breaks. And the uniform study hints that the estimates matter: SVD-LLM's second version did worse with a different rank per layer than with uniform ranks. That's what you'd expect if the sensitivities are noisy, because optimising against noisy estimates favours the layers whose errors were underestimated, a winner's curse. That's my reading; the study didn't test it. Bootstrap-stabilised sensitivities are another of my proposals.

## Improving it for efficiency

On the systems side, the improvements are more concrete. Align every compressed dimension to the hardware's tile sizes; it's free and can be the difference between a speedup and a slowdown. Prefer formats the hardware runs natively, eight-bit floating point on recent data-centre GPUs and four-bit floating point with fine-grained scales on the newest, over formats that need software decoding. Make the compression procedure itself cheaper: SliceGPT spends about three and a half hours on a single H one hundred for seventy billion parameters, with double-precision eigendecompositions a major cost that randomized methods could cut.

## What to remember

Training-free compression spans three tiers, from data-free rounding to calibrated optimisation of rotations, and fair comparisons stay within a tier. Nearly all methods share one objective: make each compressed layer reproduce its original outputs on calibration data, with the input second-moment matrix as the Hessian. Four-bit weight-only quantization is solved and is the default. Two bits is a frontier. Pruning, slicing, low-rank and depth removal all lose to quantization at equal memory, and are best used where quantization can't go, or stacked on top of it. The uniform study broadly confirmed this and added four rules: on a dense model, beat a smaller sibling or don't bother; measure decode bytes per token, not stored bytes; on a mixture-of-experts model, fewer experts per token was the cheapest lever; and stacking can fail outright, as SliceGPT did under four-bit microscaling. Finally, the statistics underneath, effective sample size, heavy tails, calibration design, uncertainty and allocation, are mostly unaddressed, which makes them, in my judgement, a promising place to improve.

## Self-check questions

Ten questions, with brief answers.

One. What are the three tiers of training-free compression? Data-free, calibration-only with closed-form computation, and calibrated optimisation of auxiliary parameters such as rotations, with the weights frozen.

Two. What objective do most methods share, and what matrix sits at its heart? Matching each compressed layer's outputs to the original on calibration data; the second-moment matrix of the layer's inputs, which acts as the Hessian.

Three. Why are activations harder to quantize than weights, and how do rotations help? A few channels carry huge outliers that swamp a shared four-bit scale; an orthogonal rotation spreads them across all channels, while computational invariance keeps the network's output unchanged.

Four. Why can a two-bit seventy-billion model lose to a four-bit seven-billion one? With some older two-bit methods, the quantization error outweighs the bigger model's advantage, even though the bigger model uses far more memory; gentle compression of a smaller model can win.

Five. Why don't rotations help NVIDIA's four-bit floating-point format much? Its scale is shared by only sixteen values, and the scale itself is finer, so the scaling already absorbs most of the outlier problem, and a rotation can spread outliers into groups that had none.

Six. What does GPTAQ change about GPTQ? It calibrates each layer on the inputs it will actually receive from the already-quantized earlier layers, so it corrects propagated errors, at almost no extra cost.

Seven. Why might the bottom of an activation spectrum be unreliable? Tokens within a sequence are correlated, so the effective sample size may be far below the token count, and for slowly varying directions close to the sequence count, where small eigenvalues and their directions are mostly noise.

Eight. Why can a narrower matrix fail to run faster? If its new dimension isn't aligned to the GPU's tile sizes, the kernel falls back to slower paths and the speedup can vanish or reverse.

Nine. In the head-to-head study, why is decode bytes per token a better axis than stored bytes? At small batch, decoding is limited by memory traffic. Removing or merging experts shrinks storage, but if the router still picks the same number of experts per token, the bytes read per token don't change.

Ten. On a small dense model, what baseline must a structured or low-rank method beat, and did any? A smaller model from the same family. None did at this scale: the best methods paid the same or worse perplexity for far less compression.

That's the state of training-free compression. Thanks for listening.
