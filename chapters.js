/**
 * The library. One object per lecture — this is the only place chapter data
 * lives.
 *
 * It used to be spread across three: hand-written <li> markup in index.html, a
 * parallel BACKS array in index.js matched purely by position, and hand-typed
 * chapter numbers. Adding a lecture meant editing two files and keeping the
 * indices lined up, and getting it wrong silently gave a chapter the wrong
 * artwork. Now adding a lecture is one object here, and the number is just its
 * position in this array.
 *
 *   title — shown in the sidebar and the player
 *   src   — the audio file; omit for a reading-only chapter
 *   doc   — optional markdown transcript for the reader pane
 *   bg    — artwork behind the player
 */
const CHAPTERS = [
  {
    title: 'Splines: Concepts and Applications',
    src:   'combined.mp3',
    bg:    'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=1920&q=80',
  },
  {
    title: 'An Introduction to Machine Learning',
    src:   'combined2.mp3',
    bg:    'https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1920&q=80',
  },
  {
    title: 'An Introduction to Non Linearity',
    src:   'combined3.mp3',
    bg:    'https://images.unsplash.com/photo-1545987796-200677ee1011?w=1920&q=80',
  },
  {
    title: 'Longitudinal Modelling',
    src:   'combined4.mp3',
    bg:    'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1920&q=80',
  },
  {
    title: 'Spline Interpretation',
    src:   'combined5.mp3',
    bg:    'https://images.unsplash.com/photo-1509228468518-180dd4864904?w=1920&q=80',
  },
  {
    title: 'Trees',
    src:   'combined6.mp3',
    bg:    'https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=1920&q=80',
  },
  {
    title: 'SVM',
    src:   'combined7.mp3',
    bg:    'https://images.unsplash.com/photo-1558618666-fcd25c85f82e?w=1920&q=80',
  },
  {
    title: 'Neural Networks',
    src:   'combined8.mp3',
    bg:    'https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1920&q=80',
  },
  {
    title: 'The Rashomon Effect',
    src:   'rashomon_combined.mp3',
    bg:    'https://images.unsplash.com/photo-1478720568477-152d9b164e26?w=1920&q=80',
  },
  {
    title: 'Rashomon Analysis & Model Instability',
    src:   'paper1_combined.mp3',
    bg:    'https://images.unsplash.com/photo-1494232410401-ad00d5433cfa?w=1920&q=80',
  },
  {
    title: 'Ensemble Learning & Gradient Boosting',
    src:   'boosting_combined.mp3',
    bg:    'https://images.unsplash.com/photo-1527474305487-b87b222841cc?w=1920&q=80',
  },
  {
    title: 'Decomposed Random-Effects Tree',
    src:   'decomposed_tree_combined.mp3',
    bg:    'https://images.unsplash.com/photo-1518186285589-2f7649de83e0?w=1920&q=80',
  },
  {
    title: 'Transformers & LLM Design Decisions',
    src:   'transformer_combined.mp3',
    doc:   'readable_docs/transformer_explainer.md',
    bg:    'https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1920&q=80',
  },
  {
    title: 'How Embedder Models Are Trained',
    src:   'embedder_combined.mp3',
    doc:   'readable_docs/embedder_training.md',
    bg:    'https://images.unsplash.com/photo-1639322537228-f710d846310a?w=1920&q=80',
  },
  {
    title: 'Thinking in the Dark: Recurrent Depth Models',
    src:   'recurrent_depth_combined.mp3',
    doc:   'recurrent-depth-models-lecture.md',
    bg:    'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=1920&q=80',
  },
  {
    title: 'Making Ouro Fast: Hardware & Kernel Optimization',
    src:   'hardware_ouro_combined.mp3',
    doc:   'hardware-ouro-lecture.md',
    bg:    'https://images.unsplash.com/photo-1591405351990-4726e331f141?w=1920&q=80',
  },
  {
    title: 'Four Models, One GPU: A Hardware Comparison',
    src:   'model_comparison_combined.mp3',
    doc:   'model-comparison-lecture.md',
    bg:    'https://images.unsplash.com/photo-1518770660439-4636190af475?w=1920&q=80',
  },
  {
    title: 'Ouro in Practice: Implementation Plan',
    src:   'ouro_implementation_combined.mp3',
    doc:   'ouro-implementation-lecture.md',
    bg:    'https://images.unsplash.com/photo-1515879218367-8466d910aaa4?w=1920&q=80',
  },
  {
    title: 'The Model Is No Longer Just the Model: Serving in August 2026',
    src:   'inference_serving_combined.mp3',
    doc:   'inference-serving-lecture.md',
    bg:    'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1920&q=80',
  },
  {
    title: 'Same Model, Twice as Fast: Optimizing for Throughput',
    src:   'throughput_combined.mp3',
    doc:   'throughput-optimization-lecture.md',
    bg:    'https://images.unsplash.com/photo-1504639725590-34d0984388bd?w=1920&q=80',
  },
];
