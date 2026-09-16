# Rashomon Analysis & Model Instability

Paper 1. When Many Models Fit Equally Well: Rashomon Analysis, the Noise-Simplicity Connection, and a Taxonomy of Model Instability for Psychology.

Target: Psychological Methods or Psychological Review.

Executive Summary: Three Contributions.

First, psychological outcomes are noisy. Noisy outcomes produce large Rashomon sets, as shown by Semenova and others in 2023. Large Rashomon sets mean many models fit equally well. Therefore, single-model reporting in psychology is formally expected to be inadequate. This is not an empirical observation but a theoretical prediction.

Second, model instability comes in two types: epistemic and representational, with different consequences. This taxonomy is novel.

Third, jointly examining feature-level instability using MCR and predictive multiplicity provides an empirical diagnostic for partially separating the two types.

## Section 1. Introduction: The Problem Is Theoretically Predictable

Machine learning methods have been rapidly adopted in psychology. Studies report a single best model and derive conclusions from it. We argue this is formally expected to fail. The argument has three steps, each grounded in existing theory.

Step 1, from Semenova and others 2023: Outcome noise expands the Rashomon set. Noisier data leads to a flatter loss surface, which leads to more near-optimal models. This is not a conjecture; it is a proven mathematical result.

Step 2, an empirical fact: Psychological outcomes are among the noisiest in science. Self-report measures have test-retest reliabilities of 0.7 to 0.85. Behavioral outcomes have substantial day-to-day variability. Signal-to-noise ratios are typically low.

Step 3, the logical consequence: Psychology should exhibit large Rashomon sets. Many structurally different models will achieve near-identical performance. Conclusions drawn from any single model are one of many equally valid explanations.

This three-step argument makes a specific, falsifiable prediction: the Rashomon ratio for typical psychological datasets should be large. We test this prediction empirically in Section 6. If confirmed, the implication is stark: the single-model reporting convention is not just a practical weakness but a theoretically predicted failure mode for noisy applied domains.

## Section 2. The Rashomon Framework: Formal Foundations

## Section 2.1. The Rashomon Set

Let F be a function class, ell a loss function, S a dataset, and f star the empirical risk minimizer. The Rashomon set, as defined by Fisher and others in 2019, is: R of epsilon, F, and S equals the set of all f in F such that L of f on S is less than or equal to L of f star on S plus epsilon.

## Section 2.2. The Rashomon Ratio

The Rashomon ratio, from Semenova and others 2022, is the volume of the Rashomon set relative to the hypothesis space: R ratio of F and theta equals the volume of R divided by the volume of F. It ranges from 0 to 1. A large ratio means many models achieve near-optimal performance. The Rashomon ratio is fundamentally different from standard complexity measures such as VC dimension and Rademacher complexity. It depends on both the function class and the dataset, capturing the specific degree of model ambiguity for the problem at hand.

Cheap diagnostic: Semenova and others showed that if several different ML algorithms achieve similar performance on a dataset, this is evidence of a large Rashomon ratio. Before building the full Rashomon set, researchers can check: do linear regression, random forest, XGBoost, and a GAM all achieve roughly the same test loss? If yes, the Rashomon ratio is likely large, and single-model conclusions are likely fragile. This check costs minutes, not hours.

## Section 2.3. Model Reliance and Model Class Reliance

Fisher and others in 2019 defined Model Reliance, or MR, for a model f as the ratio of f's expected loss when covariate X 1 is permuted to its loss without permutation. They proved that MR estimators are U-statistics, enabling finite-sample inference. This is their Theorem 5. This gives the diagnostics a rigorous statistical foundation. MR is not a heuristic but an estimator with known bias, variance, and convergence properties.

Model Class Reliance, or MCR, extends MR across the Rashomon set: MCR minus equals the minimum of MR of f, and MCR plus equals the maximum of MR of f, over all f in the Rashomon set R of epsilon. The MCR range width, which is MCR plus minus MCR minus, is the primary estimand of feature-level instability. Fisher and others provide finite-sample bounds for MCR using covering numbers.

## Section 2.4. The MR-Causal Connection

Fisher and others, in their Proposition 19, proved that for binary covariates, MR can be written as a function of conditional causal effects of X 1 on Y. This is not merely an analogy. It is a formal mathematical relationship. For binary treatment indicators, clinical group assignments, and diagnostic categories, which are common in psychology, model reliance directly encodes causal effect information. This substantially strengthens the argument that Rashomon stability serves as indirect evidence of structural plausibility: when MR is mathematically linked to causal effects, MCR minus greater than zero means that every near-optimal model encodes a non-zero causal-adjacent quantity.

## Section 2.5. Rashomon Importance Distribution, or RID

Donnelly and others at NeurIPS 2023 addressed MCR's instability under resampling by bootstrapping the dataset B times, computing the Rashomon set for each, and averaging importance CDFs. RID has proven exponential convergence via Hoeffding's inequality. The HIV viral load application discovered the gene LINC00486 as robustly important, a finding that single-model importance and even MCR without bootstrapping missed. This serves as a template for what Rashomon analysis could discover in psychological data.

## Section 2.6. The Noise-Simplicity Theorem

This is a key theorem for psychology. Semenova and others in 2023 proved that Rashomon sets constructed from noisy data tend to contain simpler models than corresponding sets from non-noisy data. Noise expands the set of good features and enlarges the set of models using at least one good feature. For psychology, this means: the noisier your outcome measure, and psychological outcomes are very noisy, the more models fit it well, and the more likely that a simple interpretable model is among them. The complexity premium of boosted models over linear models is theoretically expected to be small for noisy psychological outcomes.

## Section 2.7. The Covering Argument: Why Simpler Models Exist

Semenova and others in 2022 proved: if a simpler function class F 1, for example sparse linear models, serves as a delta-cover for a complex class F 2, for example all boosted models, meaning for every f in F 2 there exists g in F 1 within delta in prediction space, then a sufficiently large Rashomon set in F 2 must contain models from F 1. Rudin and others in 2024 noted that any boosted decision tree is equivalent to a single tree of greater depth, so sparse trees naturally cover boosted ensembles. This provides formal mathematical justification for testing whether simpler model classes fall within the Rashomon threshold. It is not a heuristic. It is a consequence of the covering number structure of the hypothesis spaces.

## Section 3. Two Types of Model Instability. This is the novel theoretical contribution

The distinction between epistemic and representational instability has not been formalized in the Rashomon literature. Fisher and others discuss MCR ranges. Donnelly and others discuss importance stability. RashomonGB discusses predictive multiplicity. None distinguish why models in the Rashomon set disagree. This section provides that distinction and proposes an empirical diagnostic using the joint examination of MCR and predictive multiplicity.

## Section 3.1. Epistemic Instability

The data genuinely underdetermines the model. Different near-optimal models produce different predictions for the same individuals and different feature importance rankings. Both interpretations and decisions are unstable. Epistemic instability arises when signal-to-noise ratio is low, sample size is moderate, and the predictive surface admits multiple functional forms.

## Section 3.2. Representational Instability

Different models encode the same underlying function through different parameterizations. They produce similar predictions but differ in which features they credit. This arises when the function class is over-parameterized, features are correlated enabling substitution, or the model class permits multiple decompositions of the same joint surface.

## Section 3.3. The Two-by-Two Diagnostic Matrix

The two types can be partially distinguished by jointly examining feature-level instability, measured by MCR range width, and predictive multiplicity, the established term from RashomonGB for disagreement in individual predictions across the Rashomon set.

When predictive multiplicity is low and MCR range is narrow, the situation is stable. Low instability overall. Single-model reporting is adequate.

When predictive multiplicity is high and MCR range is narrow, this is unusual. Models agree on importance but disagree on predictions. This may indicate sensitivity to outliers.

When predictive multiplicity is low and MCR range is wide, this indicates representational instability. Explanations differ but decisions are robust. Follow-up should use functional distance analysis.

When predictive multiplicity is high and MCR range is wide, this indicates epistemic instability. Both explanations and decisions are fragile. This is the most concerning case. Follow-up requires more data or accepting ambiguity.

## Section 3.4. Consequences Differ by Type

For epistemic instability: feature importance rankings are unstable, individual predictions are unstable, risk classifications are unstable, clinical decisions are undermined, scientific interpretation is undermined, and the required follow-up is more data, a stronger design, or accepting ambiguity.

For representational instability: feature importance rankings are unstable, but individual predictions are stable, risk classifications are stable, clinical decisions are likely safe, scientific interpretation is still undermined, and the required follow-up is functional distance metrics, where a prediction-level audit may suffice.

## Section 4. Rashomon Stability as Indirect Evidence of Structural Plausibility

This argument is stronger than previously framed. Fisher and others' Proposition 19 proves that for binary covariates, MR is a function of conditional causal effects. This is not an analogy. It is a theorem. When X 1 is binary, such as a treatment indicator, clinical group assignment, or diagnostic category, MCR minus greater than zero means every near-optimal model encodes a non-zero causal-adjacent quantity.

For continuous covariates, the connection is weaker. MR reflects predictive, not causal, contribution. But the logic still applies directionally: genuinely causal mechanisms impose functional constraints that should be harder to substitute away than noise artifacts.

Boundary conditions: The connection is strongest for binary covariates, via the formal causal connection in Proposition 19. It is moderate for continuous covariates with low-to-moderate inter-predictor correlation, meaning rho less than 0.4. It weakens as correlation increases, since substitution enables real effects to appear dispensable. And it is not a substitute for causal identification strategies, but rather a complement to them.

## Section 5. Formal Comparison to Existing Stability Approaches

Compared to bootstrap importance, stability selection, multiverse analysis, and specification curve analysis, Rashomon analysis uniquely provides: feature instability assessment, predictive multiplicity measurement, cross-model comparison, cross-class comparison via the covering argument, partial epistemic versus representational distinction, and formal inference since MR is a U-statistic with known finite-sample properties.

## Section 6. Empirical Demonstration

## Section 6.1. Step 0: The Rashomon Ratio Diagnostic

Before any formal Rashomon analysis, apply Semenova and others' cheap diagnostic: fit 4 to 5 structurally different ML algorithms, such as linear regression, regularized regression, random forest, XGBoost, and GAM, on the same psychological dataset. If they achieve similar test performance, the Rashomon ratio is likely large. Report the inter-algorithm performance range as the first empirical result. If this range is small, for example delta R squared less than 0.03, the single-model convention is immediately suspect.

## Section 6.2. Rashomon Ratio Estimation

Using depth-bounded decision trees as a surrogate, following Semenova and others 2022, estimate the Rashomon ratio directly. Compare the estimated ratio for the psychological dataset to the ratios reported by Semenova and others for their benchmark datasets, providing context for whether the psychological data exhibits an unusually large or small Rashomon effect.

## Section 6.3. The Instability Taxonomy in Practice

Construct the within-class Rashomon set for XGBoost. Compute MCR range width for each variable and predictive multiplicity, including individual prediction range and classification disagreement rate, across the Rashomon set. Apply the two-by-two diagnostic matrix. Target: identify at least one variable in each cell, stable, representationally unstable, and epistemically unstable.

## Section 6.4. TreeFARMS Calibration

On a simplified version of the problem using sparse decision trees of bounded depth, apply TreeFARMS from Xin and others 2022 for exact Rashomon set enumeration. Compare MCR ranges from the exact set to those from the approximate retraining-based set. This quantifies the undercoverage bias of the approximation, the degree to which retraining underestimates the true range of model reliance. This calibration step is unique to this paper and provides the first empirical measurement of approximation quality for a psychological dataset.

## Section 6.5. The Bootstrap-Stable but Rashomon-Unstable Case

Demonstrate at least one feature whose importance is stable under bootstrap resampling, seed perturbation, and stability selection, but whose MCR range is wide. This is the core justification for Rashomon analysis beyond existing methods.

## Section 7. Limitations of Paper 1

First, the epistemic-representational distinction is partially separable. The two-by-two matrix is a diagnostic heuristic, not a formal decomposition.

Second, the MR-causal connection in Proposition 19 holds exactly only for binary covariates. For continuous covariates, the link is directional, not formal.

Third, the noise-simplicity theorem predicts large Rashomon sets for noisy data but does not guarantee that the specific simple models psychologists want, meaning interpretable and clinically meaningful models, will be among them.

Fourth, TreeFARMS calibration is limited to sparse decision trees. It validates the approximation for one function class, not all.
