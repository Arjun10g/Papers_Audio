# How Embedder Models Are Trained

*A step-by-step walkthrough of the pipeline that turns a pretrained transformer into a semantic search workhorse.*

---

## Opening Framing

When you use a chat model like GPT or Claude, you're talking to a **generative transformer**. But quietly running alongside those flashy models is a whole other class of transformer — the **embedder**. Embedders are the workhorses of semantic search, retrieval-augmented generation, clustering, duplicate detection, and almost any task where you need to compare two pieces of text by meaning rather than by exact words. And they're trained in a way that's quite different from how generative models are trained.

---

## 1. What an Embedder Actually Is

An embedder takes a piece of text — anywhere from a single word to a long document — and produces a **fixed-length vector** of numbers. The vector is usually somewhere between 300 and 4,000 numbers long. That's it. No next token, no generated text, no probability distribution. Just one vector per input.

The magic isn't in the vector itself. It's in what the **vector space** means. An embedder is trained so that texts with similar meanings produce nearby vectors, and texts with different meanings produce distant vectors. You measure similarity with **cosine similarity** (the angle between two vectors) or **dot product**.

Once you have that, remarkable things become possible:

- Embed a user question, compare against embedded documents, find the most relevant.
- Cluster a million customer reviews by topic without defining the topics.
- Detect duplicates across different phrasings.

The embedder is a machine that **compresses meaning into geometry**.

---

## 2. The Core Training Goal

The goal is simple:

> Given two pieces of text that ought to be similar, make their vectors close. Given two pieces of text that ought to be different, make their vectors far apart.

Every technical decision in embedder training is in service of this one idea.

---

## 3. Architecture: From Tokens to a Single Vector

Almost every modern embedder **starts from a pretrained transformer** — usually an encoder-style one like BERT, though decoder-style transformers are increasingly used too. The pretrained model already knows grammar, word meanings, and context. We're teaching it to produce useful summary vectors.

A transformer outputs one vector **per token**. We need a single vector per input. So we **pool**:

| Pooling method | How it works |
|---|---|
| **CLS pooling** | Use the representation of the special `[CLS]` token (BERT's original approach) |
| **Mean pooling** | Average all output token vectors — often works surprisingly well |
| **Max pooling** | Take the element-wise max across tokens |
| **Attention pooling** | A small learned mechanism weights the tokens |

After pooling, many embedders add a small **projection head** (a couple of linear layers) to map to the final embedding dimension. Finally, the output is **L2 normalized** (divided by its own length) so every vector has magnitude 1. Once vectors are unit length, dot product and cosine similarity become the same thing.

---

## 4. Contrastive Learning: The Core Training Trick

The dominant training technique is **contrastive learning**. You need:

- **Positive pairs** — two pieces of text that belong together semantically
- **Negative pairs** — two pieces of text that don't

### Where positive pairs come from

You can't get them from humans at scale, so researchers mine them in the wild:

- **Q&A sites** (Stack Exchange, Reddit): question + accepted answer
- **Wikipedia**: article title + opening paragraph
- **Research papers**: title + abstract, or citation pairs
- **Search logs**: query + click (gold standard — a human thought it was relevant)
- **Products**: title + description

### In-Batch Negatives

Instead of curating negatives explicitly, use the **other examples in the training batch**. With a batch of 128 positive pairs, for any given pair the positive is its partner and the 127 other second-elements serve as negatives. You get 127 negatives for free per positive.

---

## 5. The InfoNCE Loss

The standard contrastive loss is **InfoNCE** (noise contrastive estimation), also called **NT-Xent** (normalized temperature-scaled cross-entropy).

For each anchor query:
1. Compute its similarity to every candidate in the batch.
2. Divide by a **temperature parameter** τ.
3. Apply softmax.
4. The loss is the negative log probability of the correct positive.

In plain language: *"Here's a query. Which of these 128 candidates is its true partner? Pick one."*

### Temperature Matters

| τ value | Effect |
|---|---|
| **0.02 – 0.05** (low) | Peaky softmax — forces the model to discriminate between similar candidates |
| **0.1 – 0.2** (higher) | Softer softmax — more forgiving, trains stably, less final sharpness |

Production embedders typically use **τ ∈ [0.02, 0.1]**. One of the most important hyperparameters.

### Why Batch Size Is Huge

Because negatives come from inside the batch, **bigger batch = more negatives per step = stronger training signal**. Small batches (64–128) work okay. Large batches (8K–64K) produce dramatically better embedders. This is why top embedder models train on large clusters with aggressive memory optimization.

---

## 6. The Multi-Stage Training Pipeline

Training a modern embedder is not one run. It's a **pipeline of stages**.

### Stage 1: Weakly Supervised Contrastive Pretraining

- **Data**: hundreds of millions to billions of noisy pairs mined from the open web
- **Sources**: Reddit, Stack Exchange, Wikipedia, academic papers, product catalogs
- **Duration**: hundreds of thousands of steps, batch sizes in the thousands
- **Result**: a model that knows the broad shape of semantic space

The pairs are noisy. Many are imperfect. But the dataset is huge and diverse, so the model learns the general contours of meaning.

### Stage 2: Supervised Fine-Tuning

- **Data**: much smaller (few hundred thousand to a few million), but cleaner
- **Popular datasets**: MS MARCO (Bing query-document), Natural Questions (Google), SNLI, MultiNLI
- **Result**: specific task competence — e.g., excellent passage ranking for queries

Pretraining gave general semantic sense. Fine-tuning gives specific behavior.

### Stage 3: Hard Negative Mining (Iterative)

In-batch negatives are cheap but often **too easy**. A Python question vs. a lasagna recipe is trivial to distinguish. The informative negatives are the **almost-right** ones:

- Python question paired with a JavaScript answer
- Medical query paired with a veterinary document

**How to mine them:**

1. Use the current embedder.
2. For each query in the training set, retrieve the top 100 most similar documents.
3. The true positive is somewhere in the top results. Everything else in the top 100 is a plausible-but-wrong match — a hard negative.
4. Add them to training. Retrain.
5. Repeat 3–4 times. Each round the model's judgment of "hard" gets sharper.

Top embedders (BGE, E5, Jina) all use multi-round hard negative mining as a core part of their recipe. Recent work also uses LLMs to **generate synthetic hard negatives**.

### Stage 4: Task-Specific Fine-Tuning (Optional)

If you know the domain (legal docs, scientific papers, code), one final tune on a small amount of in-domain data produces significant gains.

---

## 7. Evaluation: MTEB and Its Subtleties

The dominant benchmark is **MTEB** (Massive Text Embedding Benchmark). It aggregates dozens of tasks:

- Retrieval
- Classification
- Clustering
- Reranking
- Pair classification
- Summarization similarity
- Semantic textual similarity (STS)

### The Key Insight

**These tasks don't all want the same thing.**

| Task | What it wants |
|---|---|
| Retrieval | Precisely calibrated query-document matching |
| Classification | Embeddings that linearly separate class labels |
| Clustering | Tight, intuitive groups |
| STS | Fine-grained similarity judgments between pairs |

There is no single best embedder. An embedder amazing at retrieval might be merely okay at clustering. Modern releases often include several sizes and variants — small fast, big accurate, specialized.

---

## 8. Modern Wrinkles

### Matryoshka Embeddings

Train the model so that the **first N dimensions** of the output are themselves a valid, shorter embedding. A model produces a 4000-D vector, but the first 2000 work, and the first 1000 work, and the first 500 work — all the way down. Done by computing the contrastive loss on truncated versions as well as the full vector and averaging.

Result: **trade quality for speed at inference time by just truncating the vector**. Huge win for production, where storage and compute scale linearly with dimension.

### Instruction-Tuned Embeddings

The same text might need to be embedded differently depending on the downstream task. Instruction-tuned embedders accept a natural-language prefix:

> `"Represent this sentence for retrieving related Wikipedia articles: climate change"`

The model uses the instruction to shape the embedding. One model usable across many tasks without retraining. Pioneered by the **E5** and **Instructor** families.

### Late Interaction (ColBERT)

All the embedders above produce **one vector per document**. ColBERT keeps **one vector per token**. At query time it matches each query token to its best document token and sums.

| | Single-vector | ColBERT (late interaction) | Cross-encoder |
|---|---|---|---|
| Expressiveness | Low | Medium | High |
| Storage | Small | Large | N/A (scores at query time) |
| Query speed | Fast | Medium | Slow |

ColBERT sits in the middle ground between fast single-vector retrieval and slow-but-accurate cross-encoder reranking.

---

## 9. Tying It Together

We started with a goal: train a model that maps text to vectors where similar meanings are close and different meanings are far.

We took a pretrained transformer → added pooling → trained with **contrastive learning** → used **positive pairs** from naturally occurring sources → used **in-batch negatives** scaled by a **temperature parameter** → ran a **multi-stage pipeline** (weak pretraining → supervised fine-tune → hard negative mining → optional task tune) → evaluated on **MTEB** → layered on **Matryoshka**, **instruction tuning**, and **late interaction** as needed.

Every step was a decision. Every decision a tradeoff:

| Decision | Tradeoff |
|---|---|
| Bigger batch | Better negatives ↔ more memory |
| Harder negatives | Better discrimination ↔ expensive to mine |
| Larger dimension | Richer embeddings ↔ more storage |
| Instruction tuning | Task flexibility ↔ more complexity |

The next time you use semantic search, RAG, or any system that compares text by meaning, there's an embedder underneath doing the work — and it got good at its job through this exact pipeline: a pretrained language model, taught by contrast, refined by mining, evaluated against a dozen notions of "good."

That's how you train a machine to understand meaning well enough to put it on a map.
