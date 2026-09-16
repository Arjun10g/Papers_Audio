# The Rashomon Effect

*The Rashomon Effect in Machine Learning*

## The Core Idea

The term comes from Leo Breiman in 2001, who named it after Akira Kurosawa's 1950 film Rashomon, where four witnesses give contradictory but equally plausible accounts of the same event. The Rashomon Effect describes the phenomenon that there exist many equally good predictive models for the same dataset, and when it happens, it sparks both magic and consternation, but mostly magic.

The formal definition is straightforward. Given a loss function ell, a reference model f star (typically the best-performing one), and a threshold epsilon, the Rashomon set is all models f in your function class F whose loss is within epsilon of the best.

The Rashomon set R of epsilon and F equals all models f in F such that the loss of f is less than or equal to the loss of f star plus epsilon.

This sounds simple, but the implications are profound for the original question about underfitting and complexity gaps.

## Why This Matters for the Linear Versus Boosting Question

When you fit a linear model and a boosted model and see a performance gap, the natural question is: is that gap real signal, or could a simpler model close it? The Rashomon perspective reframes this. Semenova, Rudin, and Parr hypothesize that there is an important reason simple yet accurate models often exist: the Rashomon set is often large, and if it's large, it contains numerous accurate models, and perhaps at least one of them is the simple model we desire. Their key result is that problems where the outcome is uncertain or noisy tend to admit large Rashomon sets and simpler models, which has significant policy implications as it undermines the main reason for using black box models for decisions that deeply affect people's lives.

So rather than asking how much extra variance does boosting capture, the Rashomon perspective asks how large is the set of near-optimal models, and does it contain something interpretable? If it does, the complexity gap might be an artifact of search, not a fundamental feature of the data.

## The Key Tools That Have Been Developed

Model Class Reliance, or MCR, by Fisher, Rudin, and Dominici in 2019. This framework describes how much any model class, any model-fitting algorithm, or any individual model relies on covariates of interest. The key quantity is a range: MCR minus is the minimum reliance across all good models, MCR plus is the maximum. A feature with a large MCR minus is important in all well-performing models; a feature with a small MCR plus is unimportant to every well-performing model. This directly connects to the question: if the nonlinear interaction terms have low MCR minus, meaning some good models don't need them at all, then the extra variance your boosting model captures via interactions might not be essential.

TreeFARMS, by Xin, Zhong, Chen, and others in 2022. This provides the first technique for completely enumerating the Rashomon set for sparse decision trees, in fact the first complete enumeration of any Rashomon set for a non-trivial problem with a highly nonlinear discrete function class. Before this, people could talk about Rashomon sets theoretically but couldn't actually look inside them. TreeFARMS changed that, enabling three concrete applications: studying variable importance across all near-optimal trees, translating Rashomon sets across different metrics such as accuracy to F1, and examining stability under data subsetting.

The Rashomon Importance Distribution, or RID, by Donnelly, Katta, Rudin, and Browne, presented as a NeurIPS 2023 Spotlight. This addresses a critical flaw in existing variable importance methods. For a given dataset, there may be many models that explain the target outcome equally well; without accounting for all possible explanations, different researchers may arrive at many conflicting yet equally valid conclusions given the same data. The specific problem they identified is that even MCR is unstable across bootstrap iterations. For a given variable, one bootstrap might suggest it's completely unimportant while another suggests it's essential to all good models. RID fixes this by averaging the importance distribution across bootstrapped Rashomon sets, producing stable estimates. They demonstrated its utility with a case study exploring which genes are important for predicting HIV load, highlighting an important gene called LINC00486 that had not previously been studied in connection with HIV.

RashomonGB, presented at NeurIPS 2024. This paper systematically analyzes the Rashomon effect specifically for gradient boosting, providing rigorous theoretical derivations and an information-theoretic characterization of the Rashomon set for boosting algorithms. This is directly relevant to the setup, since you can now formally characterize how many equally good boosted models exist and how much their reliance on interactions varies.

## The Amazing Things Perspective by Rudin and Colleagues at ICML 2024

The most comprehensive statement of the research program is the 2024 ICML paper by Rudin and colleagues. They address how the Rashomon Effect impacts: first, the existence of simple yet accurate models; second, flexibility to address user preferences such as fairness and monotonicity without losing performance; third, uncertainty in predictions, fairness, and explanations; fourth, reliable variable importance; fifth, algorithm choice; and sixth, public policy.

The theoretical backbone is an elegant covering argument: larger Rashomon sets tend to contain multiple simpler models because for every model in the more complex space, there exists a close model from the simpler space. Sparse decision trees serve as a cover for deeper, more complex decision trees, and trees are universal function approximators. So if your Rashomon set is large, which it tends to be for noisy real-world problems, a simple model almost certainly exists within it.

## The Interactive Paradigm

What makes this practical rather than purely theoretical is the tooling. Instead of finding one optimal model, the algorithms find many good models and visualize them, so the user can interact with them to choose among them. This is the Rashomon set paradigm. Tools like TimberTrek let domain experts browse hundreds of millions of near-optimal decision trees, filtering by properties they care about such as fairness constraints, specific feature inclusions or exclusions, and monotonicity. Even if the Rashomon set contains hundreds of millions of models, when they are organized effectively, humans can navigate them in real time.

## How This Connects Back to the Original Question

Here's the synthesis. The original question was about decomposing the variance gap between a linear model and a complex model. The Rashomon perspective adds a crucial dimension: is that gap stable? Specifically:

Step one. Fit your linear model and your boosted model. Observe the R squared gap.

Step two. Compute the Rashomon set for the boosted model class, meaning all boosted models within epsilon of optimal.

Step three. Within that Rashomon set, compute MCR for the interaction terms. If MCR minus for interactions is near zero, there exist near-optimal boosted models that barely use interactions, meaning the complexity premium is fragile.

Step four. Check whether the Rashomon set contains models from your simpler class, such as linear or GAM. If it does, the gap isn't structural, it's a search artifact.

Step five. Use RID to check whether these conclusions are stable across bootstrap resamples.

This gives you not just how much variation does complexity capture, but how robust is that finding, and could a simpler model do almost as well, which is arguably the more important applied question.

The key papers to read in order would be: Semenova and others 2022 on existence of simpler models, Fisher and others 2019 on MCR, Donnelly and others 2023 on RID, and Rudin and others 2024 Amazing Things as the synthesis piece.

## Applied Workflow Example

Let's say your dataset is something like predicting depression severity using a PHQ-9 score from a battery of psychological, demographic, and behavioral predictors, including sleep quality, rumination, social support, age, income, childhood adversity, exercise frequency, and so on. Maybe 500 to 1,000 participants with 15 to 20 predictors. This is typical of clinical psychology research.

Step 1: Establish the Complexity Gap. Fit your linear regression, either OLS or regularized, and your gradient boosted model such as XGBoost or LightGBM. Use proper cross-validation. Suppose you get R squared linear equals 0.38 and R squared boost equals 0.46. That's an 8 percentage point gap. The traditional interpretation would be that nonlinearity and interactions account for 8 percent of additional variance in depression scores. But is that robust?

Step 2: Define the Rashomon Set. Choose a loss threshold epsilon. A common choice is a percentage of the best model's loss, say models within 5 percent of the best test loss. For your boosted model with MSE equals 12.4, you'd include all boosted models with MSE less than or equal to 13.02. Generate the Rashomon set by retraining with different hyperparameter configurations including learning rate, max depth, subsample ratio, number of trees, and regularization, as well as different random seeds. Collect every model that falls below the threshold. In practice, you might retrain 500 to 1,000 times and keep those that qualify. The RashomonGB framework from the NeurIPS 2024 paper provides a more principled way to characterize this set for boosting specifically.

Step 3: Characterize Structural Complexity Within the Rashomon Set. For each model in the Rashomon set, measure its effective complexity. For tree-based models, useful proxies include average tree depth, where shallow trees approximate main effects and deeper trees approximate higher-order interactions; the number of splits that involve more than one feature along a path, which serves as an interaction proxy; and SHAP interaction values aggregated to get the total variance attributable to interactions versus main effects. Plot the distribution: what fraction of near-optimal models are simple, meaning shallow with low interaction reliance, versus complex? If most near-optimal boosted models are shallow with minimal interaction terms, your 8 percent gap is likely inflated. Much of it can be recovered with nonlinear main effects alone using a GAM, not interactions.

Step 4: Compute Model Class Reliance for Specific Interaction Terms. Suppose the single best boosted model suggests a rumination times sleep quality interaction drives predictions. Compute model reliance using permutation importance for that interaction across every model in the Rashomon set. If MCR minus is near zero, meaning some near-optimal models don't rely on that interaction at all, you cannot confidently claim this interaction is a real structural feature of the data. It might be a quirk of one particular fit. Conversely, if MCR minus is high, every good model needs it, and you can be more confident it's genuine.

Step 5: Test Whether Simpler Model Classes Fall Within the Rashomon Set. This is the critical test. Fit a series of increasingly simple models and check whether any of them achieve loss below your Rashomon threshold of MSE less than or equal to 13.02. First, a GAM with nonlinear main effects and no interactions, which isolates whether the gap is about nonlinearity versus interactions. Second, a GAM with the top 2 to 3 detected interactions, which tests whether a small number of interactions closes the gap. Third, a linear model with polynomial terms for the most nonlinear features. Fourth, a plain linear model. If the GAM alone falls inside the Rashomon set, your conclusion changes dramatically: the complexity gap between linear and boosted models is primarily due to nonlinear main effects, not interactions. A GAM with interpretable shape functions is statistically indistinguishable from the best boosted model. If even the linear model falls inside, the gap was essentially noise. The boosted model was fitting to sampling variability, and a linear model is defensible.

Step 6: Stability Analysis via the Rashomon Importance Distribution. The Rashomon set from a single dataset can be unstable. Apply the RID framework: bootstrap your 800 participants, say 200 times, recompute the Rashomon set for each bootstrap, and compute the variable importance distribution for each predictor across all Rashomon sets across all bootstraps. This gives you a full probability distribution over importance values. For your interaction of interest, rumination times sleep, if the RID shows most of its mass near zero, the interaction is not a stable finding. If it shows a clear mode away from zero with tight spread, it's robust.

Step 7: Report and Interpret. Your results section in a psychology paper would now look fundamentally different from the standard approach of using XGBoost and SHAP. Instead, you could report something like: The best boosted model achieved R squared equals 0.46 compared to R squared equals 0.38 for the linear model. However, examination of the Rashomon set, containing 517 models within 5 percent of optimal loss, revealed that 73 percent of near-optimal models relied minimally on interaction terms, with interaction-attributed variance less than 1 percent. A GAM with nonlinear main effects achieved R squared equals 0.44, falling well within the Rashomon set. Model Class Reliance analysis showed that the rumination times sleep interaction had MCR minus equals 0.002, indicating it is dispensable in nearly all good models. The Rashomon Importance Distribution confirmed that the importance of this interaction was not stable across bootstrap resamples. We conclude that the apparent complexity gap is driven primarily by nonlinear dose-response relationships in individual predictors, particularly a threshold effect in sleep quality below 5 hours, not by interactions. A GAM is sufficient and preferred for interpretability.

Step 8: The Practical Payoff for Psychology. This matters enormously in clinical psychology for a few specific reasons. First, treatment targeting: if interactions are real, you'd tailor treatment differently for different subgroups, for example a sleep intervention specifically for high ruminators. If interactions are artifacts, subgroup-specific treatment protocols are unjustified. Second, replication: psychology's replication crisis is partly driven by unstable findings from single models on noisy data. The Rashomon approach explicitly quantifies this instability. Third, clinical deployment: a GAM can be turned into a clinician-facing scoring tool. A boosted model with 500 trees cannot.

## What You'd Need Computationally

For tree-based Rashomon sets, the TreeFARMS algorithm handles sparse decision trees exactly. For boosted models, the practical approach is the retraining strategy, varying hyperparameters and seeds systematically. The InterpretML package from Microsoft gives you Explainable Boosting Machines that naturally decompose into main effects and interactions. The SHAP library provides interaction values. The Rashomon Importance Distribution GitHub repository from Donnelly and others implements RID directly.

The whole pipeline is feasible with standard computing resources for a dataset of the size typical in psychology. It adds maybe a day of computation and a week of analysis time, but the inferential gains are substantial. You go from boosting beats linear by 8 percent to a nuanced understanding of where that 8 percent comes from and whether it's real.
