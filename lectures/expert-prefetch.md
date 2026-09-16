# What We Learned About Predictive MoE Prefetching

### A lecture on the September 2026 expert-prefetch session

*Approx. 30 minutes spoken. Written to be listened to: the few numbers that matter are rounded and said in words, the mathematics is described rather than written, and the run records, file names, and exact figures live in the written session notes rather than here. The headline is deliberately modest. The predictors got better at guessing, but on the workload we measured they did not beat a plain reactive cache.*

---

## The question

Welcome back. Today's subject is a systems experiment with a negative headline and a lot of useful detail underneath it. The question is whether a mixture-of-experts language model can predict which experts it is about to need, early enough to move their weights into place before they are used.

Here is the setup. A mixture-of-experts model contains many expert feed-forward networks, but each token is routed to only a small handful of them. In the smaller of the two models we studied, a token chooses eight experts out of two hundred and fifty-six. The experts that are not chosen may live in host memory, or in some slower tier, and so the runtime would love to fetch the experts it is about to need while the graphics card is busy doing useful work on the current layer.

There are two possible sources of advantage here, and it matters that we keep them apart. The first is prediction: can we identify the future experts before the model's own router reaches that layer? The second is lead time: can we issue those transfers early enough that the bytes actually arrive before the expert is consumed?

Those are genuinely different problems. A predictor can be statistically accurate and still lose, if computing its guess costs more time than the head start it creates. And the reverse is also true: a weak signal that arrives a whole token early can be worth more than a strong signal that shows up a few microseconds before it is needed. So the session measured route quality, predictor cost, cache traffic, transfer behaviour, and end-to-end token time as five separate quantities, and refused to collapse them into one number.

## A short chronology

The session came in two parts. The first part closed out an earlier experiment on the older thirty-billion-parameter model. In that run, the corrected predictor was applied to the second half of the network, and the earlier layers kept the incumbent predictor.

The numbers from that closeout set the tone for everything after. The router-based prefetcher and the corrected prefetcher were essentially tied: the corrected one was about a quarter of a percent faster per token, and the uncertainty around that figure comfortably included zero. Both of them, though, were clearly faster than the plain reactive cache, the arm with no prediction at all, which was roughly a sixth slower per token. So the experiment supports a large gap between reactive caching and router-based prefetching on that workload, but it does not establish a confident extra speedup from the correction.

The second and main part of the session expanded the evidence for two newer models, which I will call the smaller model, Qwen three point five, and the larger model, Qwen three point eight. We captured the smaller model's hidden states, fit correction artifacts on some requests, and selected them on others that the fit had never seen. We replayed several predictor variants on the larger model. We measured predictor heads on a data-centre graphics card. We confirmed how transfer copies behave. And we ran three end-to-end settings on the smaller model, proposing eight, twelve, and sixteen candidate experts per token per layer.

## Routing, and where the prediction error lives

Let me set up the idea in words, because one decomposition explains the whole predictor programme.

At any given layer, the model's router looks at the token's hidden state and produces one score per expert. The experts with the highest scores are the ones that run. Prefetching does not change this decision, ever. The router remains the authority at the moment the expert is consumed. Speculation only proposes bytes to stage early.

Now suppose we want to use an earlier layer to predict the route at a later layer. The deployed predictor, which keeps no extra state at all, does the simplest possible thing: it takes the later layer's router and applies it to the earlier layer's hidden state. We call that score the anchor.

The true score at the later layer differs from the anchor by an increment: it is whatever the router would add once the network has transformed the state between the two layers. So the real score is the anchor plus that increment.

This is the key to everything that follows. Cross-layer predictors can see the anchor. What they cannot see is the increment, the change the network will make between the two layers, and that unseen increment is where their error lives. So every method in the session can be read as an attempt to estimate some useful piece of that increment, subject to two conditions: the estimate has to carry over from the training requests to requests it has never seen, and its compute has to fit inside the lead-time budget.

## Native experts versus proposed candidates

Two ideas are easy to confuse, so let me separate them explicitly. One is the native count: the number of experts the model actually consumes for a token, eight in the smaller model and ten in the larger. The other is the candidate breadth: the size of the shortlist we hand to admission and cache filtering. The breadth is not the number of transfers that reach the hardware. Experts already resident, duplicate guesses, byte budgets, and admission rules can all shrink the actual transfers well below the shortlist.

When the shortlist is exactly as long as the native count, it has to be perfectly right to contain the complete native set. Making the shortlist longer raises the chance of covering all the native experts, but it can increase the bytes admitted and lower the precision of each guess.

One consequence is worth stating up front, because it looks like a bug in the results and is not. The larger model consumes ten experts per token. Therefore a shortlist of eight can never contain all of them; full coverage at breadth eight is zero by definition. That does not mean an eight-wide predictor is useless. Some of the ten may already be sitting in the cache, so the question that matters is how many of the *missing* experts it catches. The report keeps those measures separate, and so will I.

## The timing pipeline

It helps to picture the runtime as a timeline. Early in the token, at the source layer, we make a prediction. That prediction goes through admission and a cache filter, and the surviving candidates are issued as speculative transfers. Their bytes then become progressively ready. Meanwhile, and independently, the model keeps progressing through its layers until its own router runs at the target layer. At the moment the expert is needed, one question is asked: are its bytes ready? If yes, we compute. If no, we fall back to a reactive fetch, wait, and then compute.

Two words in that description are doing careful work. Predicted means an expert was proposed. Ready means its bytes had arrived, in the usable layout, by the time they were needed. The router does not wait for speculative bytes; model progress and speculation proceed independently, and a transfer that arrives after the router has spoken but before the expert actually runs can still help. A predicted expert can be too late, partially ready, evicted, or simply unnecessary. The implementation tracks readiness separately from prediction, so that a high accuracy number cannot be mistaken for hidden transfer time.

The available lead may be one layer, several layers, or a whole token. A later source layer has a better state but less time. An earlier source layer has more time but a less faithful state. This is why the session ranked methods by lead, and measured each target layer on its own, rather than collapsing all layers into one predictor score.

## The metrics, and why they are kept apart

Here are the measures, and the reason each one exists.

Recall is the fraction of the experts the model actually used that appeared on the shortlist.

Precision is the fraction of the shortlist that turned out to be used.

Full coverage is stricter than recall: did the shortlist contain every native expert, not just most of them?

Cache-aware miss coverage is the fraction of experts that were *not* already in the cache which the shortlist correctly proposed in time. Of all these, it is the measure closest to what we actually want, which is avoiding reactive fetches.

Issues and waste count the transfers actually issued, and the ones that turned out not to cover a miss. Waste can rise even as recall rises.

Readiness is whether an issued transfer had arrived by the time it was needed.

And finally the traffic ratio: all the transfers the prefetcher caused, including the reactive fetches it failed to prevent, divided by the transfers a plain reactive cache would have made. A ratio below one would mean the prefetcher moved fewer bytes than the reactive cache. Recall alone cannot tell you this.

To see why the separation matters, imagine a sixteen-wide shortlist with excellent recall that drags in many unneeded experts, next to an eight-wide shortlist with lower recall that covers a useful share of the native set with far fewer bytes. Quality and traffic have to sit side by side. The report never converts offline quality into an unmeasured speedup.

## Predictor families

The incumbent is just the anchor: the later router applied to the earlier state. The session screened several ways to correct it.

The first is bias correction. On the training data, look at the increment the network added between the two layers, and take its average across examples. Add that average to the anchor. This captures a systematic direction shared across tokens. It costs one vector addition and can be folded into the score computation almost for free.

The second is velocity correction. The hidden state has just moved from the previous layer to the current one. Project that recent movement through the later router, and ask whether the direction the state just moved predicts the direction it will move next. A single scalar, fit by least squares, decides how much of that velocity to add. It is cheap, but on requests it had not seen, a fixed velocity correction was not reliably better than the incumbent.

The third, and the main candidate for the smaller model, is a low-rank drift correction. It learns a small number of directions in the earlier state that correlate with the increment, and applies a correction only along those directions. The number of directions is called the rank. Three implementation forms have to be distinguished, because they cost different amounts.

The algebraic form computes the anchor, projects the state onto the learned directions, and applies the correction as explicit extra work.

The augmented form appends the correction features to the score computation itself. That saves some launch overhead, but the projection work does not disappear.

The precomposed form folds the correction into an effective dense weight and bias, so that the corrected score costs exactly one matrix multiply, the same as the incumbent. This is only possible when the correction reads the same input state as the anchor. The effective weight keeps the router's shape; the arm simply owns an extra precomposed copy alongside the native router weights.

None of these runs a separate full router at the target layer, and the hardware comparison keeps the three paths distinct.

The larger model adds a wrinkle. Its layers pass along a wide carrier state, four streams side by side, which each layer's mixer narrows down to the router's input. So the code tested several ways to read that carrier before applying the target router. Source mix runs the carrier through the *earlier* layer's mixer. Target mix runs it through the *later* layer's mixer, which uses more target-specific information, but reads the full carrier and pays the mixer's cost. Anchor uses the ordinary narrow router input, as in the deployed proxy. Carrier keeps the full wide state as the correction basis. Because that basis is four times wider than the router's input, it cannot be folded into the same precomposed head, and a mixer is a nonlinear transformation whose runtime work stays on the bill.

One phrase to hold on to: zero stored artifact. The target-mix arms use mixer weights that are already resident, so they add no persistent weights of their own. But the state they read is wider and the runtime mixer work still matters. Zero stored bytes does not mean zero cost.

## Low-rank correction, step by step

Let me walk through the fit in words, because the structure is what makes the precomposition trick work.

Take a batch of training tokens at one pair of layers. For each token, project the increment, the change in state between the two layers, through the later router. That gives the increment in score space.

Take the average of those score increments across the batch. That is the bias. Subtract it out, so what remains is the part of the increment that varies from token to token.

Now centre the earlier states as well, and ask: which directions in the earlier state move together with the remaining score increment? Form the cross-covariance between the two and take its leading singular directions. Keep only the top few. That handful of directions is the learned basis, and how many you keep is the rank.

Project each earlier state onto those directions, and fit a small regularised least-squares map from the projected coordinates to the score increment. That map is the correction matrix. At inference, the corrected score is the anchor, plus the bias, plus the projected state times the correction matrix.

Each rank is fitted independently; a rank-sixty-four correction is never a truncated rank-one-twenty-eight one, even though the nested bases share their leading directions.

Now the precomposition trick. When the correction reads the same state as the anchor, the whole expression, anchor plus bias plus projected correction, is linear in the state. So the terms that multiply the state can be collected into one effective weight, and the constant terms into one effective bias. The corrected head becomes one dense multiply, the same shape and the same cost as the incumbent router. That is why a same-basis correction can be precomposed, and why a correction that reads the wide carrier cannot be quietly treated as if it read the narrow anchor.

Two safeguards mattered. First, a full high-capacity fit can memorise the drift of the particular requests it was trained on and then fail on a new request; the constrained low-rank fit is what keeps it honest, together with selection on requests the fit never saw. Second, the session ran a shuffled control: break the pairing between each state and its increment, refit, and see whether the same structured correction survives. If it does, it is an artifact, not evidence that the earlier state predicts the drift.

One more detail that prevents a double-counting mistake. The fitted correction is damped by a shrinkage factor, three quarters for the smaller model. The runtime then applies the artifact as is, because the selected artifact already contains that shrinkage. Applying the factor twice would be wrong.

## Model geometry and the selected configurations

An expert's weights, in half precision, come to about six megabytes in the smaller model and about nine and a half megabytes in the larger one. The smaller model has two hundred and fifty-six experts per layer across forty layers and uses eight per token; the larger has five hundred and twelve experts across forty-eight layers and uses ten.

As a concrete memory check, holding eight of the smaller model's experts for a single token at a single layer is nearly fifty megabytes before any cache metadata, alignment, or staging. That arithmetic is a payload example, not a measured transfer time, and not a claim about how many candidates admission will actually issue.

The smaller model's offline capture recorded a few hundred decode steps per request after a warm-up. Three requests were used for fitting, two for validation, and three held back for testing, with warm-up rows excluded throughout. Selection used validation recall, averaged per request, with the smaller rank breaking ties. The chosen policy was rank one hundred and twenty-eight at breadth eight, and rank sixty-four at breadths twelve and sixteen, all at shrinkage three quarters. That is a global policy across layers; it is not a claim that every layer independently chose the same rank.

The larger model's replays used two requests as reciprocal folds, about twenty target layers in the second half of the network, and compared source mix, target-mix anchor, and target-mix carrier. Its rank-sixty-four target-mix result is descriptive, not a deployment choice.

The full tuning grid screened four ranks, three shrinkages, leads of one, two, and four layers, and two cache sizes. The deployed headline remains lead one. And, importantly, measured cost was not used to select the frozen rows. Cost is reported afterwards, to decide whether a quality choice is practical.

## Offline results

Let me tell the offline results as a story rather than a table.

For the smaller model, the incumbent predictor already recalls about four fifths of the native experts with an eight-wide shortlist, about nine tenths with twelve, and a little more with sixteen. The learned correction lifts each of those by a percentage point or two, and lifts full coverage by a few points more, without ever making traffic worse. So the correction helps, consistently and modestly.

But look at the traffic. Even at breadth eight, the prefetcher moves about one and a half times the bytes of the plain reactive cache. At breadth sixteen it moves more than three times as many. The shortlist buys recall with bytes.

For the larger model, all at rank sixty-four, target mix is clearly the strongest family. With a sixteen-wide shortlist it recalls about ninety-five percent of the native experts and fully covers the set about two thirds of the time. Reading the full carrier instead of the narrow anchor matches that quality almost exactly, but no better. Source mix trails well behind, and the router-only controls sit below their corrected counterparts. And remember that at breadth eight, full coverage for this model is zero by definition, because it uses ten experts per token.

The lesson repeats: even the best larger-model row, at ninety-five percent recall, still moves about twice the bytes of the reactive cache. That ratio is why quality alone cannot justify a runtime claim.

Three cautions about the offline numbers. First, the offline capture was generated with sampling, while the end-to-end workload uses deterministic greedy decoding; the two are separate samples and must not be pooled. Second, the model's own routing decisions are the ground truth. Reconstructing the route from raw scores can disagree with the model because of precision conversions and tie-breaking, so the traces keep the model's native decisions, and a diagnostic oracle that peeks at the future state is used only as a ceiling, never deployed. Third, a stock reference implementation of the model produced slightly different output tokens for reasons that were never established, so it was reported descriptively and excluded from the paired comparisons. Exact token identity was confirmed among all six staged arms; it was not confirmed against that reference.

## Hardware and transfer measurements

The hardware side has three pieces: predictor microbenchmarks, transfer measurements, and the end-to-end runs, which get their own section.

The predictor microbenchmarks time the predictor head on its own, on a single data-centre graphics card, with the model itself not running. For the smaller model, the precomposed correction costs the same as the incumbent, roughly seventy microseconds per call including synchronisation, and the algebraic two-stage version costs about ten percent more. So precomposition really does make the correction free at the head.

For the larger model the picture is starker. The incumbent costs about seventy-five microseconds. Anything that runs the target mixer costs half again as much or more, well over a hundred microseconds, and owns several times as many bytes, because the mixer's work stays in those rows. Older microbenchmarks that did not charge the mixer cannot be reused as final results. And to repeat the scope: these are measurements of the predictor head alone. They do not include the model and do not establish end-to-end token speed.

On methodology, each timing row is many repeated calls within a single invocation, not that many independent observations, and the report uses medians across invocations.

Now transfers. An exploratory screen varied host threads, transfer streams, and payload layout: packed and contiguous versus split into tiles. The confirmatory run then fixed one host thread, one stream, and one tile size, and compared three copy strategies at three candidate sizes on both models, over thirty paired rounds.

The result is narrow but clean. Batching the candidate experts into one contiguous copy was a few percent faster than copying them one by one, between one and four percent depending on model and breadth. Splitting the copies into tiles was slower, by about twelve percent. For scale, a baseline copy of eight of the smaller model's experts took just under a millisecond.

And here is the caveat that keeps this honest. Those payloads were synthetic byte buffers sized like expert weights. They were not actual expert gathers from the model. The result supports a narrow copy-path observation, not a model speedup claim. Also, coalescing in this context means merging software transfer commands, which is a different thing from the memory coalescing that happens inside the graphics card. The replay merged almost none of the commands, a handful out of tens of thousands, and no on-card coalescing gain was measured.

## End-to-end results

This is the section the whole session was built to reach. Three end-to-end runs on the smaller model tested shortlists of eight, twelve, and sixteen. Each run compared three arms: the plain reactive cache, router-based prefetching, and the corrected, precomposed prefetcher.

The first finding: router-based prefetching lost to the reactive cache, and it lost by more as the shortlist grew. About five percent slower per token at breadth eight, about seven and a half at twelve, and nearly nine at sixteen.

The second finding: the corrected prefetcher and the router-based one were tied. The differences were a tenth of a percent here and a few tenths there, sometimes in the corrected arm's favour, sometimes not, on three requests. Real perhaps, but tiny, and specific to this workload.

Which means the third finding follows directly: every corrected arm was also slower than the reactive cache, by the same five to nine percent.

Precomposition did matter. The un-folded version of the correction was about four percent slower than router prefetching at every breadth, and up to thirteen percent slower than the reactive cache. Folding the correction into the dense head removes most of that extra cost. It just does not turn a loss into a win.

Meanwhile the engine's own counters confirm that the quality gain is real: in-engine recall for the corrected arm was a point or so higher than the router's at every breadth. Yet the corrected arm fetched between about ten percent and seventy-five percent more bytes than the reactive cache, rising with breadth. Those counters are software tallies, not bus measurements, and they do not imply that the predictor improved token latency.

One of the three runs, at breadth twelve, showed noticeable drift between its two passes; it is reported descriptively rather than thrown out by a threshold invented after the fact. The earlier failed attempts remain in the evidence trail rather than being silently replaced.

And the larger model got no whole-model end-to-end claim at all. Its checkpoint is several hundred gigabytes, more than a hundred of them in non-expert weights, which does not fit the single-card harness. Its head, cache-replay, and transfer evidence are useful for design screening. They cannot be promoted to a model-level speedup.

## Engineering repairs and reproducibility

A large share of the session went into making the results auditable, and the repair chronology is part of the result. Every run carries its configuration, its seed, the exact model version, the exact source code identity, and a receipt of its results. The final analysis checks those identities, pairs rows within each request and pass, and refuses to fall back on an older run. Configurations are frozen once they have produced evidence; corrections go into new configurations. The head and transfer results were verified against remote hashes; the end-to-end lifecycle had no remote digest, so its local artifacts were hashed and checked for membership rather than described as remotely verified.

Several practical catches mattered. The stock reference implementation produced mismatched tokens and was excluded from paired comparisons. A cache-aware replay and a runtime coalescing replay have different scopes, so their counters must not be asserted equal. Predictor-owned bytes, workspace bytes, graph pools, persistent model weights, and physical bus traffic are five different memory quantities. The packed contiguous copy was verified byte-exact for its synthetic source and destination, which says nothing about an arbitrary gather until the real path is exercised. And the rented hardware and its temporary keys were deleted once the artifacts were banked.

The code repairs included teaching the capture path to read the smaller model's native routing decisions directly, rather than guessing at the structure, since a different model family lays them out differently. Half-precision exports now keep their exact bit patterns with explicit provenance. The capture path was pinned to one library version and made portable across operating systems. Coefficient exports read only frozen fits and selection metadata and are hash-checked. A resume path for interrupted scoring reads held-out states and binds the checkpoint to the frozen fit without refitting. Shared handling of the carrier, the candidates, and device-to-host copies was repaired before the final measurement.

For context on the larger model's mixer, since it matters for the cost argument: it is not a simple reshape. It normalises the four streams, projects them down, computes a gate, and averages the gated streams. That is nonlinear work, which is why target-mix cost stays on the bill and cannot be absorbed into the same linear folded head as the anchor.

Reproducibility has two levels. The final report can be regenerated from the banked compact evidence: receipts, tables, selected coefficient artifacts, source code, and provenance. Re-running the original capture and fits requires the raw hidden states, the model weights, and suitable hardware, which the archive does not contain.

## What the results mean

The session supports five conclusions.

First, predictor quality and runtime benefit are separate things. The correction improves offline recall and nudges the in-engine counters, but every measured corrected arm is still slower than the reactive cache. Extra transfer traffic and the fixed cost of issuing transfers are plausible explanations, and so is the fact that a reactive cache already captures repeated routes with a long, token-scale lead. Where exactly the time goes was not established; the attempt to watch the bus directly used invalid sampling.

Second, the strongest larger-model rows are the target-mix rows, but they pay mixer and head work and still carry a transfer ratio above one. Reading the full carrier matches the anchor's quality, yet it reads four times as much state. The state representation, the stored artifact, and the runtime compute must all be charged.

Third, rank is a hardware-and-software decision. A higher rank helps the smaller model at the narrowest shortlist, while a lower rank was selected at the wider ones. Selection used held-out recall, and measured head cost was applied afterwards to judge practicality. The right deployment decision combines held-out behaviour with measured cost, rather than treating rank in isolation.

Fourth, transfer layout matters, but the measured transfer result is narrow. A synthetic packed copy can be a few percent faster, and tiling can be substantially slower. The data path still needs a real manifest, arbitrary expert selection, coalescing, event dependencies, and safe cancellation before that number becomes a runtime claim.

Fifth, the negative result is useful. It tells us that improving a predictor's guesses is not enough when the reactive cache and the transfer schedule already dominate. The next design has to optimise the entire chain, lead time, cache state, admission, readiness, transfer commands, and native compute, together.

## What comes next

These are priorities, not completed results.

One: run a real end-to-end comparison with a noise threshold declared in advance and enough independent requests to resolve sub-percent effects.

Two: instrument the physical bus and expert readiness timestamps, so that software byte counters can be separated from actual bus movement.

Three: exercise packed and coalesced transfer on arbitrary, non-contiguous expert selections, including cancellation.

Four: measure a real dependent-fetch round trip and a priority transfer path; the synthetic screen cannot provide those.

Five: test the larger model on a sharded or multi-card harness that can hold it, preserving exact token identity and the carrier provenance.

Six: choose breadth and rank jointly under measured bandwidth slack. A method should be admitted only when its candidate bytes can arrive before the deadline.

Seven: compare source mix and target mix under a fixed predictor budget, charging mixer compute, carrier read bytes, cache state, and readiness, rather than comparing recall alone.

## Glossary

A few terms, briefly, in case any slipped past.

The anchor, or router proxy, is the early score computed by applying the target layer's router to an earlier layer's state. The carrier is the larger model's wide, four-stream state that each layer's mixer narrows down. Candidate breadth is the length of the speculative shortlist proposed per token per layer. The native count is the number of experts the model's own router actually selects. Full coverage means every native expert was on the shortlist. Lead is the distance, in layers or tokens, between where the prediction is made and where the expert is consumed. Readiness is whether the required bytes are present in a usable state when needed. A reactive miss is a required expert that is absent, or not ready, when the router asks for it. Shrinkage is a damping factor applied to a fitted correction. Rank is the number of learned state directions the low-rank correction uses. End to end means the full token workload, including model execution and the transfer path. A microbenchmark is an isolated predictor or transfer measurement with a narrower scope.

## Self-check questions

Ten questions, with brief answers after each.

One. For the larger model, why is full coverage at breadth eight zero while recall at breadth eight is still meaningful, and how does that differ from cache-aware miss coverage? Because eight guesses cannot contain ten experts; recall still measures the overlap, while cache-aware miss coverage only counts the experts that were not already resident.

Two. In the score decomposition, anchor plus increment, which part is unseen by the predictor? The increment: the change the network makes to the state between the source layer and the target layer.

Three. Why can a predictor with higher recall still increase total transfer? Extra candidates cost bytes and commands, and the cache state changes the value of each candidate.

Four. What does the low-rank fit actually learn? A small number of directions in the earlier state that move together with the score increment, and a map from those directions to a correction.

Five. Why does the carrier-reading correction need to be charged for state reads and mixer work even when it stores no extra weights? Because zero stored artifact does not mean zero reads or zero compute.

Six. Why are the offline sampled results not interchangeable with the end-to-end results? They have different sampling and workload scopes, and different units of independence.

Seven. What does a small corrected-versus-router difference on three requests establish, and what does it fail to establish? It describes a small paired effect on this workload; it is not a broad speedup proof.

Eight. Why can the packed contiguous transfer result not be reported as a model-level gain? The payload is synthetic and contiguous, unlike arbitrary expert gathers.

Nine. Which decision remains authoritative in the runtime pipeline? The model's own router and its native weights.

Ten. What additional evidence is required before the larger model can receive a whole-model end-to-end claim? A sharded or multi-card harness that fits the model, with exact provenance and paired end-to-end controls.

That is the session. The headline is negative, the evidence is careful, and the next experiment is already written down. Thanks for listening.
