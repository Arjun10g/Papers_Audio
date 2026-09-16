# What We Learned About Predictive MoE Prefetching

### A lecture on the September 2026 expert-prefetch session

*Approx. 40 minutes spoken. Written to be listened to: the numbers are rounded and said aloud, the equations are described in words, and the run identifiers, file paths, and hashes live in the written session notes rather than here. The central result is deliberately modest. The predictors improve offline route quality, but on the measured Qwen 3.5 workload they do not beat a plain reactive cache on end-to-end token time.*

---

## The question

Welcome back. Today's subject is a systems experiment with a negative headline and a lot of useful detail underneath it. The question is whether a mixture-of-experts language model can predict which experts it is about to need, early enough to move their weights into place before they are used.

Here is the setup. A mixture-of-experts model contains many expert feed-forward networks, but each token is routed to only a small handful of them. Call the total number of experts E, and the number the router actually picks K. In Qwen 3.5, a token chooses eight experts out of two hundred and fifty-six. The experts that are not chosen may live in host memory, or in some slower tier, and so the runtime would love to transfer the experts it is about to need while the GPU is busy doing useful work on the current layer.

There are two possible sources of advantage here, and it matters that we keep them apart. The first is prediction: can we identify the future expert IDs before the native router reaches that layer? The second is lead time: can we issue those transfers early enough that the bytes actually arrive before the expert is consumed?

Those are genuinely different problems. A predictor can be statistically accurate and still lose, if computing its score costs more time than the transfer slack it creates. And the reverse is also true: a weak signal that arrives a whole token early can be worth more than a strong signal that shows up a few microseconds before it is needed. So the session measured route quality, predictor cost, cache traffic, transfer behaviour, and end-to-end token time as five separate quantities, and refused to collapse them into one number.

## A short chronology

The session came in two parts. The first part closed out an earlier experiment on Qwen 3, the thirty-billion-parameter model with three billion active. In that run, the native router was top eight, the residency cache held thirty-two experts, and the corrected arm applied a rank thirty-two correction at source layers twenty through forty-six, which is twenty-seven layers. The twenty earlier source layers kept the incumbent predictor.

The numbers from that closeout set the tone for everything after. The router-based prefetch arm averaged about forty-two point zero five milliseconds per token. The corrected arm averaged about forty-one point nine five. And the reactive least-recently-used arm, the plain cache with no prediction at all, averaged about forty-eight point nine. So the corrected arm was about a quarter of a percent faster than the router arm, with a ninety-five percent interval running from roughly minus one point seven percent to plus one point two percent, over three request means. That interval straddles zero. The experiment supports a large gap between reactive LRU and router-based prefetch on its own workload, but it does not establish a confident incremental speedup from the predictor.

The second and main part of the session expanded the evidence ladder for two newer models, Qwen 3.5 and Qwen 3.8. We used a request-held-out capture from Qwen 3.5 to fit and select correction artifacts. We replayed carrier and mixer variants on Qwen 3.8. We measured predictor heads on an H100. We confirmed transfer-copy behaviour. And we ran three end-to-end breadth settings on Qwen 3.5, proposing eight, twelve, and sixteen candidate experts per token per layer.

## Routing, and where the prediction error lives

Let me set up the notation, because one decomposition explains the whole predictor programme.

At a target layer, call it t, the native router receives a hidden state, h sub t, and computes one score per expert. In simplified notation, the score vector z sub t equals the router weight W sub t times the hidden state h sub t. That vector has one entry per expert. The native route is the IDs of the largest K entries, together with the router's normalised weights. Prefetching does not change this decision, ever. The native router remains the authority at the point of consumption. Speculation only proposes bytes to stage early.

Now suppose we want to use an earlier source layer, call it s, to predict the route at the later target layer t. The deployed predictor, which keeps no extra state at all, computes what we call the anchor: A equals W sub t times h sub s. That is the target layer's router weight applied to the source layer's hidden state.

The exact target score differs from that anchor by a residual increment. Write d equals W sub t times the difference, h sub t minus h sub s. Then the true score is simply z sub t equals A plus d.

This is the key to everything that follows. Cross-layer predictors can see A. What they cannot see is d, the increment the network will add between the source layer and the target layer, and that unseen increment is where their error lives. So every method in the session can be read as an attempt to estimate some useful component of d, subject to two conditions: the estimate has to transfer from the training requests to held-out requests, and its compute has to fit inside the lead-time budget.

## K versus M

Two letters are easy to confuse, so let me separate them explicitly. K is the native number of experts the model actually consumes. M is the proposed candidate breadth: the size of the shortlist we hand to admission and cache filtering. M is not the number of transfers that reach the backend. Resident experts, duplicate IDs, byte budgets, and slack admission can all reduce the actual issues below M.

When M equals K, the proposed set has to be exactly right to contain the complete native set. Raising M increases the chance of covering all the native experts, but it can increase admitted bytes and lower candidate precision.

One consequence is worth stating up front, because it looks like a bug in the tables and is not. For Qwen 3.8, native truth is top ten. Therefore full-set coverage at M equals eight is mathematically zero: eight IDs cannot contain ten distinct native IDs. That does not mean an M-equals-eight predictor covers no cache misses. Cache-aware recall is a different question, because some of the native experts may already be resident. The report keeps those measures separate, and so will I.

## The timing pipeline

It helps to picture the runtime as a timeline. Early in the token, at the source layer, we make a prediction. That prediction goes through admission and a cache filter, and the surviving candidates are issued as speculative transfers. Their bytes then become progressively ready. Meanwhile, and independently, the model keeps progressing through its layers until the exact native router runs at the target layer. At the consumption deadline, one question is asked: is the required expert ready? If yes, we do the expert compute. If no, we fall back to a reactive miss fetch, and then compute.

Two words in that description are doing careful work. Predicted means an ID was proposed. Ready means the required bytes had arrived, in the usable layout and synchronisation state, by the expert-consumption deadline. The native router does not wait for speculative bytes; model progress and speculation proceed independently, and a transfer that arrives after the router but before the expert is consumed can still help. A predicted expert can be too late, partially ready, evicted, or simply unnecessary. The implementation tracks readiness separately from prediction, so that a high recall number cannot be mistaken for hidden transfer time.

The available lead may be one layer, several layers, or a whole token. A later source layer has a better state but less time. An earlier source layer has more time but a less faithful state. This is why the session ranked methods by lead, and measured each target layer on its own, rather than collapsing all layers into one predictor score.

## The metrics, and why they are kept apart

Here are the route metrics, and the reason each one exists.

Recall at M is the fraction of native expert IDs that appear in the candidate set: the size of the intersection of the candidates with the native set, divided by K.

Candidate precision is the fraction of proposed candidates that are native: the same intersection, divided by M. Admission and resident filtering can make actual issues smaller than M, but the offline definition is based on the proposed set.

Full-set coverage is the probability that every native ID is contained in the candidate set. It is stricter than recall.

Cache-aware miss coverage is the fraction of experts absent from the current resident cache that are correctly proposed and admitted early. Of all these, it is the metric closest to what we actually want, which is avoiding reactive fetches.

Issues and waste count the candidate issues per layer, and the issues that do not cover a cache miss. Waste can rise even as recall rises.

Readiness is whether an issued block has arrived sufficiently by the consumption deadline. A candidate that arrives after that deadline is a miss for latency purposes.

And finally the traffic ratio. The final report defines it as candidate issues plus remaining reactive misses, divided by the matched reactive misses of the baseline. A ratio below one would mean fewer total expert transfers than the reactive baseline. Recall alone cannot tell you this.

To see why the separation matters, consider an M-equals-sixteen method with ninety-five percent recall that admits many extra experts, next to an M-equals-eight method with lower recall that covers a useful fraction of the native set with far fewer proposed bytes. Quality and traffic have to sit side by side. The report never converts offline quality into an unmeasured speedup.

## Predictor families

The incumbent, also called the router proxy, is just the anchor: A equals W sub t times h sub s. The session screened several ways to correct its score.

The first is bias correction. On the training rows, compute the projected increment for each row, and take the mean. Call that mean b. The corrected score is A plus b. This captures a systematic direction shared across examples. It costs one vector add of length E, and it can be fused into an existing score epilogue.

The second is velocity correction. The source hidden state has a local change from the previous layer, h sub s minus h sub s-minus-one. Project that velocity through the target router to get v. After centring the training residual and the velocity, fit a single scalar gamma by least squares: the sum over rows of v times the residual, divided by the sum of v squared plus a small epsilon. The score becomes A plus b plus gamma times v. The method asks whether the direction the state just moved predicts the direction it will move next. It is cheap, but the held-out results show that a fixed velocity correction is not automatically better than the incumbent.

The third, and the main Qwen 3.5 candidate, is fused low-rank drift correction. It learns a small number of directions in the source state that correlate with the projected residual, then applies a rank-r correction along those directions. Three implementation forms have to be distinguished, because they cost different amounts.

The algebraic form computes the anchor, projects the centred state through the learned basis U, and multiplies by the fitted matrix Gamma, as explicit extra work.

The augmented-rows form appends the correction features to the score computation itself. That reduces launch structure, but the remaining projection and epilogue work does not disappear.

The dense, or precomposed, form folds the correction into an effective dense weight and bias. That is only possible when the correction basis uses the same input state as the anchor. The effective weight keeps the router's E-by-H shape, and the arm owns an extra precomposed copy and bias alongside the native router weights.

None of these runs a separate full target router, and the hardware comparison keeps the three paths distinct.

Qwen 3.8 adds a wrinkle. Its hidden state is a four-stream hyper-connection carrier of width ten thousand two hundred and forty, whereas the router's input is two thousand five hundred and sixty wide. So the code tested several projections from that carrier before applying the target router weight. Source mix passes the carrier through the source layer's own mixer, producing the source layer's mixed input. Target mix passes it through the target layer's mixer, which uses more target-specific information, but reads the full carrier and pays the mixer's cost. Anchor is the ordinary router input used by the deployed proxy. Carrier keeps the full ten-thousand-wide state as the correction basis. Because that basis is four times wider than the router anchor, it cannot be collapsed into the same dense folded head, and a target mixer is a nonlinear transformation whose runtime work stays on the bill.

One phrase to hold on to: zero stored artifact. The target-mix arms use mixer weights that are already resident in an expert-streaming design, so they add no persistent weight artifact. But the state read width and the runtime mixer work still matter. Zero stored bytes does not mean zero cost.

## Low-rank correction, step by step

Let me walk through the fit, with shapes, because the shapes are what make the precomposition identity work. Take a training batch at one source-target pair with n rows. The source states form a matrix of n by H. The target states are also n by H. The router weight is E by H.

Step one, project the target increment: D equals the difference of the target and source states, times the router weight transposed. D is n by E.

Step two, take the column mean of D to get the bias b, of length E, and centre the residual: R equals D minus b.

Step three, centre the correction basis. If the basis is the source state, X equals the source states minus their mean, n by H. For a Qwen 3.8 carrier correction, X can instead be n by ten thousand two hundred and forty, while the score anchor remains the narrower router input.

Step four, form the cross-covariance C equals X transposed times R, which is H by E, and take its singular vectors. Keep the first r left singular vectors as the basis U, of shape H by r.

Step five, project each centred input into that subspace: Z equals X times U, giving n by r.

Step six, solve a regularised least-squares problem for the epilogue matrix. Gamma equals the solution of Z transposed Z plus epsilon times the identity, against Z transposed R. Gamma is r by E. The implementation sets epsilon to one millionth of the trace of Z transposed Z plus one, and uses a linear solve rather than forming an inverse.

At inference, the correction is Z times Gamma, and the final score is A plus b plus Z Gamma.

Each requested rank gets its own independently fitted Gamma. A rank sixty-four Gamma is never truncated down to rank thirty-two, even though the nested bases share their leading columns.

Now the identity. For the same-basis case, with a scalar shrinkage alpha, the score is h times W transposed, plus alpha times the bracket b plus the centred h times U Gamma. Distribute, and the h terms collect: the score equals h times the quantity W transposed plus alpha U Gamma, plus alpha times the quantity b minus mu U Gamma, where mu is the mean we subtracted. So the effective weight is W transposed plus alpha U Gamma, and the effective bias is alpha times b minus mu U Gamma. The folded head still has the router's E-by-H weight shape. That is why a same-basis correction can be precomposed into one dense multiply, and why a ten-thousand-wide carrier basis cannot be silently treated as the narrower anchor.

The important statistical choice is still the constrained cross-covariance fit. A full high-capacity fit can memorise request-specific drift and then fail on a new request. As a guard, the session also ran a shuffled-pairing control with a fixed seed: break the pairing between source state and increment, refit, and see whether the same structured correction survives. If it does, it is an artifact, not evidence that the source predicts the drift.

One more detail that prevents a double-counting mistake. Offline score shrinkage multiplies the fitted correction by a scalar. Qwen 3.5 selected a shrinkage of zero point seven five. The runtime then uses a lambda of one, because the selected artifact already contains that shrinkage. Applying the factor twice would be wrong.

## Model geometry and the selected configurations

The expert payload size in bf16 comes from the three feed-forward projections: three times the hidden size times the intermediate size times two bytes.

For Qwen 3.5, the thirty-five-billion-parameter model with three billion active, that is two hundred and fifty-six experts, native top eight, forty layers, a hidden size of two thousand and forty-eight, and an intermediate size of five hundred and twelve, which works out to about six mebibytes per expert. For Qwen 3.8 Flash Next, it is five hundred and twelve experts, native top ten, forty-eight layers, a hidden size of two thousand five hundred and sixty, and an intermediate size of six hundred and forty, which is about nine point four mebibytes per expert.

As a concrete memory check, holding eight Qwen 3.5 experts for one token and one layer is forty-eight mebibytes before cache metadata, alignment, and staging. That arithmetic is a payload example, not a measured transfer time, and not a claim about how many candidates admission will actually issue.

The Qwen 3.5 offline capture sampled three hundred and eighty-four decode steps per request after sixty-four warm-up steps. Three requests were used for fitting, two for validation, and three for test, with warm rows excluded from both fitting and scoring. The end-to-end runs, by contrast, use greedy sixty-four-token continuations. The selection metric was validation request-mean recall, with the smaller rank breaking ties. The globally selected policy was rank one hundred and twenty-eight at breadth eight, and rank sixty-four at breadths twelve and sixteen, all at shrinkage zero point seven five. That is a global rank policy; it is not a claim that every layer independently chose the same rank.

The Qwen 3.8 carrier replays used two reciprocal request folds, five hundred and twelve decode steps per request with four hundred and eighty scored after a warm-up of thirty-two, nineteen target layers, and twenty stored state layers. The focused rank grid compared source mix, target-mix anchor, and target-mix carrier. The report labels the rank sixty-four target-mix result as descriptive, not as a whole-model deployment choice.

The full tuning grid screened ranks sixteen, thirty-two, sixty-four, and one hundred and twenty-eight; shrinkages of one half, three quarters, and one; leads of one, two, and four layers; and cache capacities of ninety-six and two hundred and fifty-six. The deployed headline remains lead one. And, importantly, measured cost was not used to select the frozen rows. Cost is reported afterwards, to decide whether a quality choice is practical.

## Offline results

Let me read the offline table as a story rather than a grid. Everything here is lead one, with the Qwen 3.5 replay at a cache of ninety-six and the Qwen 3.8 replay at a cache of thirty-two. For each method I will give recall, then full-set coverage, then the total-transfer ratio against the reactive baseline.

Start with Qwen 3.5. The incumbent, at breadth eight, has recall of about seventy-nine percent, full-set coverage of about fourteen percent, and a transfer ratio of about one point four nine. The selected correction at breadth eight lifts that to about eighty percent recall and about sixteen percent full-set coverage, with a slightly lower transfer ratio of about one point four five. At breadth twelve, the incumbent has about eighty-nine percent recall and fifty-three percent coverage at a ratio of two point two eight; the selected method has about ninety-one percent recall and fifty-seven percent coverage at two point two six. At breadth sixteen, the incumbent reaches about ninety-three percent recall and sixty-nine percent coverage at a ratio of three point two six; the selected method reaches about ninety-four percent recall and seventy-three percent coverage at essentially the same ratio.

So the correction helps, consistently, by a percentage point or two of recall and a few points of full-set coverage, and it never makes traffic worse. But notice the ratios. Even at breadth eight, the prefetcher is moving about one and a half times the bytes of the reactive baseline, and at breadth sixteen more than three times.

Now Qwen 3.8, all at rank sixty-four. Source mix at breadth eight has about sixty-five percent recall and a ratio of one point two three. Target-mix anchor is much stronger: about seventy-three percent recall at breadth eight with a ratio of one point one two; about ninety percent recall and forty percent full-set coverage at breadth twelve; and about ninety-five percent recall and sixty-six percent coverage at breadth sixteen, with a ratio of about two point zero one. Target-mix carrier matches target-mix anchor almost exactly on every one of those numbers. The router-only controls sit below their corrected counterparts: source-mix control at about fifty-nine percent recall at breadth eight, target-mix control at about seventy percent.

Remember that every Qwen 3.8 full-set coverage at breadth eight is zero by definition, because native truth is top ten. And notice the same lesson as before: at breadth sixteen, target-mix anchor reaches ninety-five percent recall and sixty-six percent full-set coverage, but the total-transfer ratio is still about two. That ratio is why quality alone cannot justify a runtime claim.

Two worked examples make the table concrete. For Qwen 3.8 target-mix anchor at breadth eight, a recall of zero point seven two five means about seven and a quarter native IDs per row on average, out of ten. Candidate precision is therefore seven and a quarter out of eight, about ninety-one percent. For the breadth-sixteen row, total transfer is about seven point four issues plus about zero point eight remaining misses, divided by about four point one matched reactive misses, which comes to about two point zero one. These are replay quantities, not physical bus measurements.

Three cautions about the offline numbers. First, the Qwen 3.5 offline artifact was generated with sampling, at temperature one, top-k twenty, top-p zero point nine five, with a fixed seed, while the end-to-end workload uses its own frozen greedy configuration; do not pool the two as if they were one sample. Second, native IDs are the truth. Conversions between bf16 and fp32, normalisation, and softmax tie behaviour can make a reconstructed raw-logit top-k disagree with the native IDs, so the trace keeps the native logits, scores, and indices, and a diagnostic oracle that peeks at the future target state is used only as a ceiling, never deployed. Third, the stock Hugging Face control produced output-token mismatches whose cause was not established, so it is reported descriptively and excluded from the paired comparisons. Exact token identity was established among the six staged arms, across one hundred and eight rows, three requests, and two passes; it was not established against the stock control.

## Hardware and transfer measurements

The hardware side has three pieces: predictor microbenchmarks, transfer measurements, and the end-to-end runs, which get their own section.

The environment was an H100 PCIe with one hundred and fourteen streaming multiprocessors, Torch two point six with CUDA twelve point four, and Transformers five point seventeen. The Qwen 3.5 end-to-end runs used bf16 model execution, while the predictor head ran in fp32 with TF32 disabled. Peak allocated process memory was about seventy-six gigabytes, including the resident-control scope. That is a process-memory measurement, not a bus measurement.

The predictor microbenchmarks use the CUDA-graph path, with wall time including ID readback and synchronisation, and input update time excluded. CUDA-event timing includes enqueue gaps, so it is not kernel-only time.

For Qwen 3.5 at layer one and rank one hundred and twenty-eight, the incumbent takes roughly seventy-two microseconds of wall time per call, with about twenty-nine microseconds of event time, across all three breadths. The dense folded head is essentially the same: about seventy-one to seventy-five microseconds wall, about thirty to thirty-one microseconds event. The algebraic low-rank two-stage path is slower, at about eighty microseconds wall and forty microseconds event. On owned bytes, the incumbent holds about two megabytes, the dense folded head about four point two, and the two-stage path about three point three.

For Qwen 3.8 at layer twenty-nine and rank sixty-four, the picture is starker. The incumbent takes about seventy-six microseconds. The target-router mixer control alone takes about one hundred and twelve. The low-rank two-stage path takes about one hundred and thirty. The dense folded head takes about one hundred and eighteen or nineteen. Owned bytes climb from about five megabytes for the incumbent to between thirty-one and thirty-seven megabytes for the target-mix variants, because the mixer's cost stays in those rows. Older source-mix or legacy target-mix microbenchmarks cannot be reused as final results. And to repeat the scope: these are resident replay measurements of the predictor head alone. They do not include the full model and do not establish end-to-end token speed.

On methodology, the final predictor bench had thirty invocations and six hundred and ninety-six raw timing rows. Each row is twenty rounds of sixty-four samples, which are repeated measurements within one invocation, not one thousand two hundred and eighty independent observations. The report uses medians of the invocation summaries.

Now transfers. The exploratory screen covered two hundred and sixteen cells: one, two, or four host threads; one, two, or four requested streams; packed contiguous versus tiled payloads at two hundred and fifty-six kibibytes and one mebibyte; five rounds per cell. The confirmatory run then fixed one host thread, one active CUDA stream, and a one-mebibyte tile, for eighteen cells: three strategies, three candidate sizes, two models, over thirty paired rounds.

The confirmation result is narrow but clean. On Qwen 3.5, batched contiguous copies were about three point six percent faster than baseline at breadth eight, about two point six percent faster at breadths twelve and sixteen, with tight intervals. Tiled copies were slower, by about eleven to fourteen percent. On Qwen 3.8, batched contiguous was one to two percent faster, with the breadth-sixteen interval just touching zero, and tiled was again about twelve percent slower. For scale, a baseline copy at Qwen 3.5 breadth eight took just under one millisecond, and about one point nine milliseconds at breadth sixteen.

And here is the caveat that keeps this honest. Those payloads were synthetic byte buffers sized like bf16 expert payloads. They were not actual model expert gathers. The result supports a narrow copy-path observation, not a GPU or model speedup claim. Also, command coalescing in this context means merging software transfer commands, which is a different thing from warp-level GPU memory coalescing. The CPU replay merged only two commands out of more than forty-one thousand fetches, about five thousandths of a percent, and no GPU memory-coalescing gain was measured.

## Qwen 3.5 end-to-end results

This is the section the whole session was built to reach. Three end-to-end runs tested breadths of eight, twelve, and sixteen. The router-based prefetch arm uses the same breadth as the corrected arm, and the reactive LRU arm with a ninety-six-expert cache is the baseline. In what follows, a positive effect means the first arm is slower.

Router-based prefetch versus reactive LRU: at breadth eight, the router arm was about five point one percent slower, with an interval of roughly four point three to five point nine. At breadth twelve, about seven point five percent slower. At breadth sixteen, about eight point eight percent slower. So on this workload, prefetching with the plain router proxy loses to a reactive cache, and it loses by more as breadth grows.

Corrected precomposed versus router: at breadth eight, the corrected arm was about zero point one two percent faster, with an interval from minus zero point two two to minus zero point zero two. At breadth twelve, about zero point four percent slower, with an interval spanning zero. At breadth sixteen, about zero point zero eight percent faster, with a tiny interval that excludes zero. These are real but minuscule effects.

Corrected precomposed versus reactive LRU: about five percent slower at breadth eight, about seven point nine percent slower at breadth twelve, about eight point seven percent slower at breadth sixteen.

Each interval uses three request means and the t distribution with two degrees of freedom, so the unit of inference is the request, not the token. The tiny corrected-versus-router effects at breadths eight and sixteen have intervals excluding zero, but they are small, workload-specific comparisons with n equal to three. Breadth twelve also carried substantial drift between passes, with a pooled coefficient of variation of nearly seven percent; it is reported descriptively rather than rejected by a post-hoc threshold. None of this offsets the direct measured result: every corrected arm is slower than reactive LRU.

Precomposition does matter, though. The raw augmented-rows corrected arm was about four percent slower than the router arm at every breadth, and between nine and thirteen percent slower than LRU. Folding the correction into the dense head removes most of that extra cost. It just does not create an LRU win.

Meanwhile the engine counters show the quality movement is real. In-engine recall for the corrected arm was about eighty-one percent versus eighty percent for the router at breadth eight, about ninety-two versus ninety-one at breadth twelve, and about ninety-five versus ninety-four at breadth sixteen. Yet fetched-byte totals were about one point one three, one point three nine, and one point seven five times those of the same-breadth LRU arm. These are engine software counters across six request-and-pass cells, including prefill. They are not PCIe or HBM bus traffic, and they do not imply that the predictor improved token latency.

Noise diagnostics were descriptive: pooled coefficients of variation of about zero point one five percent at breadth eight, about six point nine percent at breadth twelve, and about zero point three five percent at breadth sixteen, with no preregistered acceptance threshold. The earlier failed diagnostic attempt and failed provider attempts remain in the evidence trail rather than being silently replaced.

And Qwen 3.8 whole-model end-to-end was not claimed at all. Its checkpoint is roughly three hundred and sixty gigabytes, including more than one hundred and ten gibibytes of non-expert weights, which does not fit the current single-GPU resident harness. Its head, cache-replay, and transfer evidence are useful for design screening. They cannot be promoted to a model-level speedup.

## Engineering repairs and reproducibility

A large share of the session went into making the results auditable, and the repair chronology is part of the result. Every run has a config, a seed, a model revision, a source identity, and a result receipt. The final reducer checks the configured identity, token, artifact, and noise fields, pairs rows inside each request-and-pass cell, and refuses to fall back to an older run. Configs are frozen once they produce evidence; corrections go into new configs, and artifact-side hashes and launch records preserve the reconstruction key. The final head and transfer receipts carry remote artifact hash verification. The end-to-end lifecycle did not have a declared remote digest, so its local artifacts were independently hashed and checked for source and config membership rather than described as remotely verified.

Several practical catches mattered. The stock Hugging Face control produced token mismatches and was excluded from paired comparisons. A cache-aware replay and a runtime coalescing replay have different scopes, so their counters must not be asserted equal. Predictor-owned bytes, workspace bytes, graph pools, persistent model weights, and physical bus traffic are five different memory quantities. The packed contiguous copy was verified byte-exact for its synthetic source and destination, which says nothing about an arbitrary gather until the real manifest path is exercised. Lifecycle records bind a batch to the exact config and source, and the H100 instances and ephemeral keys were deleted once the artifacts were banked.

The code repairs included teaching the native Qwen 3.5 discovery path to recognise the actual top-k router tuple and preserve native logits, scores, and indices, rather than guessing tuple shapes, since DeepSeek's layout differs. Bf16 hidden exports now keep their bit patterns with explicit dtype provenance. The capture path was pinned to Transformers five point seventeen and made portable across file naming conventions. Fit-only coefficient exports read frozen fits and selection metadata only, and are hash-checked. An interrupted test-scoring resume reads held-out states and binds the checkpoint plus the frozen fit and selection hashes without refitting. The shared hyper-connection, candidate, and packed device-to-host handling were all repaired before the final measurement.

For context on the Qwen 3.8 carrier mixer, since it matters for the cost argument: it is not a linear reshape. With four streams, the code first applies grouped RMS normalisation and a learned weight, producing a concatenated state of width ten thousand two hundred and forty. It then computes a low-dimensional projection through a SiLU, a sigmoid gate from that projection, reshapes the state and the gate to four streams of two thousand five hundred and sixty, and averages the gated streams. That is nonlinear work, which is why target-mix cost stays on the bill and cannot be absorbed into the same linear folded head as the anchor.

Reproducibility has two levels. The final report can be regenerated from the banked compact evidence: receipts, CSVs, selected coefficient artifacts, source code, and provenance. Re-running the original capture and fits requires the excluded raw hidden-state inputs, the model weights, and suitable GPU access, which the archive does not contain. The cumulative recorded project cost estimate was about one hundred and seventy-two dollars against a two-hundred-dollar cap; that is a ledger estimate, not a bill.

## What the results mean

The session supports five conclusions.

First, predictor quality and runtime benefit are separate things. The Qwen 3.5 correction improves offline recall and slightly improves the in-engine counters, but every measured corrected arm is still slower than reactive LRU. Candidate transfer traffic and fixed issue overhead are plausible explanations, and so is the fact that LRU already captures repeated routes with a long, token-scale lead. Physical bus saturation and the exact contribution of each latency component were not established; the PCIe witness used invalid sampling.

Second, the strongest Qwen 3.8 offline rows are the target-mix rows, but they pay mixer and head work and still carry a transfer ratio above one. A target-mix carrier row can match target-mix anchor on quality, yet it reads the full carrier. The state representation, the stored artifact, and the runtime compute must all be charged.

Third, rank is a hardware-and-software decision. Rank one hundred and twenty-eight helps Qwen 3.5 at breadth eight offline, while rank sixty-four is the selected row at breadths twelve and sixteen. Selection used held-out recall, and measured head cost was applied afterwards to judge practicality. The correct deployment decision combines held-out behaviour with measured cost, rather than treating rank in isolation.

Fourth, transfer layout matters, but the measured transfer result is narrow. A synthetic packed contiguous copy can be a few percent faster, and tiling can be substantially slower. The data path still needs a real manifest, arbitrary-ID handling, coalescing, event dependencies, and consumer-safe cancellation before that number becomes a runtime claim.

Fifth, the negative result is useful. It tells us that improving a predictor's score is not enough when the reactive cache and the transfer schedule already dominate. The next design has to optimise the entire chain: lead time, cache state, admission, readiness, transfer commands, and native compute, together.

## What comes next

These are priorities, not completed results.

One: run a real Qwen 3.5 end-to-end comparison with a preregistered noise threshold and enough independent requests to resolve sub-percent corrected-versus-router effects.

Two: instrument physical PCIe and HBM traffic and expert readiness timestamps, so that software speculative-byte counters can be separated from actual bus movement.

Three: exercise packed and coalesced transfer on arbitrary manifest-selected expert blocks, including non-contiguous IDs and cancellation.

Four: measure a real dependent-fetch round trip and a priority DMA path; the synthetic screen cannot provide those.

Five: test Qwen 3.8 with a sharded or multi-GPU harness that can hold the full model, preserving exact token identity and the carrier-mixer provenance.

Six: re-evaluate breadth and rank jointly under measured bandwidth slack. A method should be admitted only when its candidate bytes can fit before the consumer deadline.

Seven: compare source mix and target mix under a fixed predictor budget, charging mixer compute, carrier read bytes, cache state, and readiness, rather than comparing recall alone.

## Glossary

A few terms, briefly, in case any slipped past.

The anchor, or router proxy, is the early score computed by applying the target router's weight to an earlier layer's state. The carrier is Qwen 3.8's four-stream hyper-connection state, ten thousand two hundred and forty wide. Candidate breadth, M, is the number of speculative expert IDs proposed per token per layer. Native K is the number of experts the model's exact router selects. Full-set coverage is the probability that every native ID is in the candidate set. Lead is the distance between the prediction source and the target consumption point. Readiness is whether the required bytes are present in a consumer-usable state. A reactive miss is a required expert that is absent, or not ready, when native routing asks for it. Shrinkage is a scalar damping of a fitted correction; Qwen 3.5 selected three quarters. Rank r is the number of learned state directions used by the low-rank correction. End to end means the full token workload, including model execution and the transfer path. A microbench is an isolated predictor or transfer measurement with a narrower scope.

## Self-check questions

Ten questions, with brief answers after each.

One. For Qwen 3.8, why is full-set coverage at breadth eight zero while recall at eight is still a valid metric, and how does that differ from cache-aware miss coverage? Because eight proposed IDs cannot contain ten native IDs; recall still measures the overlap, while cache-aware miss coverage conditions on which IDs are already resident.

Two. Write the score decomposition: the target score equals the anchor plus the projected increment. Which term is unseen by the anchor? The increment, the router weight applied to the difference between the target and source states.

Three. Why can a predictor with higher recall still increase total transfer? Extra candidates cost bytes and commands, and the cache state changes the value of each candidate.

Four. In the low-rank fit, what are the shapes of D, U, Z, and Gamma? D is n by E, U is H by r, Z is n by r, and Gamma is r by E.

Five. Why does target-mix carrier quality require charging state-read bytes and mixer work even when stored artifact bytes are zero? Because zero persistent artifact does not mean zero reads or zero compute.

Six. Why are the Qwen 3.5 offline sampled rows not interchangeable with the end-to-end workload rows? They have different sampler and workload scopes, and different units of independence.

Seven. What does a negative corrected-versus-router interval establish, and what does it fail to establish with three requests? It describes a small paired effect on this workload; it is not a broad speedup proof.

Eight. Why can the packed contiguous transfer result not be reported as a model-level GPU gain? The payload is synthetic and contiguous, unlike arbitrary expert gathers.

Nine. Which native decision remains authoritative in the runtime pipeline? The exact native router and its native weights.

Ten. What additional evidence is required before Qwen 3.8 can receive a whole-model end-to-end claim? A sharded or multi-GPU harness that fits the model, with exact provenance and paired end-to-end controls.

That is the session. The headline is negative, the evidence is careful, and the next experiment is already written down. Thanks for listening.
