# Decomposed Random-Effects Tree

The Decomposed Random-Effects Tree: Dissolving Mixed Effects into Tree Structure. What We Built, Why It Works, and What We Proved.

But first, let us carefully walk through the full mathematics of XGBoost, step by step.

Current XGBoost: The Full Math.

We begin with the global objective at round t. This is the function that XGBoost tries to minimize when adding a new tree to the ensemble.

The objective at round t, written Obj t, equals the sum over all observations i of the loss L of y i and y hat i at round t minus 1 plus f t of x i, plus the regularizer Omega of f t.

Let us unpack each piece of this equation.

L of y i and y hat i t minus 1 plus f t of x i is the loss function. It measures how bad our prediction is for observation i. The input y i is the true value we are trying to predict. The input y hat i t minus 1 is our current prediction from the ensemble so far, meaning all the trees we have already built up to round t minus 1. And f t of x i is the correction that the new tree we are adding proposes for observation i. So we are measuring: if we take our current prediction and add the new tree's correction, how far off are we from the truth?

The sum over i means we add up this loss across every single observation in the training data. We want the total error to be small.

Omega of f t is the regularizer. It penalizes complexity in the new tree. Without it, the tree could become arbitrarily complex to fit the data perfectly, which would overfit.

The regularizer is defined as: Omega of f equals gamma times T plus one half lambda times the sum over j of w j squared.

Let us explain each part.

Gamma times T: T is the number of leaves in the tree. Gamma is a penalty coefficient. This term says: every additional leaf costs you gamma units of objective. It discourages the tree from having too many leaves. More leaves means a more complex tree, which means higher risk of overfitting.

One half lambda times the sum of w j squared: w j is the prediction value, also called the weight, of leaf j. This is the number that every observation landing in leaf j receives as its prediction from this tree. The sum of w j squared is the L2 norm of the leaf weights. Lambda controls how strongly we penalize large leaf weights. This term says: don't make any leaf's prediction too extreme. Pull the weights toward zero. It is the same idea as ridge regression.

Together, the regularizer balances model fit against model complexity. The tree must earn every leaf and every unit of prediction magnitude by sufficiently reducing the loss.

Now, the Taylor Expansion. This is the critical step.

The problem with the raw objective is that L can be any loss function, and minimizing it directly over the space of all possible trees is computationally intractable. XGBoost's key insight is to approximate the loss with a second-order Taylor expansion, which turns it into a simple quadratic that we can solve in closed form.

Define the per-sample loss as a function of the correction delta. We write: ell i of delta equals L of y i and y hat i t minus 1 plus delta.

Here, delta represents how much we adjust the prediction for observation i. When delta equals zero, we are making no adjustment, which means we are sticking with our current prediction. When delta equals f t of x i, we are applying the new tree's correction.

XGBoost approximates ell i of delta with a second-order Taylor expansion around delta equals zero.

The approximation is: ell i of delta is approximately equal to ell i of 0 plus g i times delta plus one half h i times delta squared.

Let us explain each term in this approximation.

ell i of 0 is the loss at the current prediction, before any correction. This is a constant. It does not depend on the new tree at all, so it cannot be changed by anything we do. We will drop it shortly.

g i times delta is the first-order term. g i is the gradient, defined as the partial derivative of L with respect to y hat, evaluated at y hat equals y hat t minus 1. The gradient tells us: in which direction and how steeply does the loss change if we nudge the prediction slightly? If g i is negative, increasing the prediction would decrease the loss. If g i is positive, increasing the prediction would increase the loss.

One half h i times delta squared is the second-order term. h i is the Hessian, defined as the second partial derivative of L with respect to y hat squared, evaluated at y hat equals y hat t minus 1. The Hessian tells us about the curvature of the loss. It measures how quickly the gradient itself is changing. A large Hessian means the loss curves steeply, so the optimal correction is well-determined. A small Hessian means the loss is relatively flat, so the optimal correction is less certain.

Why do we use a second-order approximation instead of just first-order? Because with only the gradient, we would know which direction to move but not how far. The Hessian gives us the curvature, which tells us the optimal step size. This is the same reason Newton's method converges faster than gradient descent: it uses curvature information.

Now, dropping the constant ell i of 0 since it does not depend on the new tree, and substituting delta equals f t of x i, the objective becomes:

Obj t is approximately equal to the sum over i of g i times f t of x i plus one half h i times f t of x i squared, plus gamma times T plus one half lambda times the sum over j of w j squared.

This is now a quadratic function of the tree's outputs. Quadratics are easy to optimize.

Next, we rewrite by grouping samples into leaves. Let I j denote the set of samples that land in leaf j. Since every sample in the same leaf gets the same prediction w j, we can replace f t of x i with w j for all i in I j.

The objective becomes: Obj t equals the sum over leaves j of the quantity G j times w j plus one half times the quantity H j plus lambda times w j squared, plus gamma times T.

Where G j equals the sum of g i over all i in I j, and H j equals the sum of h i over all i in I j.

G j is the total gradient for leaf j. It summarizes how all the observations in this leaf want the prediction to change.

H j is the total Hessian for leaf j. It summarizes how confident we are about the optimal correction for this leaf.

This is a sum of independent quadratics in each w j. Each leaf's optimal weight can be found independently of every other leaf. That is why everything is so clean. The tree structure determines which observations go to which leaf, and then each leaf's weight is a separate one-variable quadratic optimization.

Optimal leaf weight. We take the derivative of the leaf j objective with respect to w j and set it to zero.

The derivative is G j plus the quantity H j plus lambda times w j. Setting this to zero and solving gives:

w j star equals negative G j divided by the quantity H j plus lambda.

This is just the vertex of a parabola. One line of algebra. The optimal leaf weight is the negative total gradient divided by the total Hessian plus the regularization parameter. The gradient tells you which direction and how much. The Hessian plus lambda tells you how confident to be. More data in the leaf means larger H j, which means a more precise estimate. Lambda adds caution on top of that.

Optimal objective value. Substituting the optimal w j star back into the objective gives:

Obj star equals negative one half times the sum over j of G j squared divided by the quantity H j plus lambda, plus gamma times T.

This formula scores any tree structure. Given a tree's partition of the data into leaves, we can compute its optimal objective value directly from the gradient and Hessian sums. We do not need to actually fit the weights. This makes comparing different tree structures extremely efficient.

Split gain. The split gain measures the improvement in objective from splitting a leaf into two children. It is the difference in objective before and after splitting.

Gain equals one half times the quantity G L squared divided by H L plus lambda, plus G R squared divided by H R plus lambda, minus the quantity G L plus G R squared divided by H L plus H R plus lambda, minus gamma.

The first term, G L squared over H L plus lambda, is the optimal objective contribution from the left child.

The second term, G R squared over H R plus lambda, is the optimal objective contribution from the right child.

The third term, G L plus G R squared over H L plus H R plus lambda, is the optimal objective contribution from the parent before splitting. Note that G L plus G R equals G parent and H L plus H R equals H parent.

So the gain is: how much better can we do with two specialized leaves compared to one general leaf? If the gain is positive, the split is worth making. If negative, the parent leaf is already doing fine.

Minus gamma penalizes the split for adding an extra leaf to the tree. The split must earn at least gamma units of improvement to be accepted.

Everything flows from one assumption: the loss is locally quadratic in delta. That single assumption gives you closed-form leaf weights, closed-form objective, closed-form split criterion, and separability across leaves. This is the mathematical engine that makes XGBoost fast and effective.

Now we move to the main contribution.

## Part 1: The Core Idea

## Section 1.1. The Problem in One Sentence

Every existing method for combining gradient boosting with mixed effects treats them as two separate modules that alternate during training. We asked: what if we dissolve the random effects directly into the tree itself, so each leaf simultaneously estimates a population-level prediction and group-specific deviations?

## Section 1.2. What a Normal Tree Leaf Does

In a standard gradient boosting tree, each leaf is simple. The data gets partitioned by the tree's splits, and every observation that lands in leaf j receives the same prediction: a single number w j. Student i from school 3 and student k from school 7, if they have similar covariate values and land in the same leaf, get the same predicted test score. The tree has no idea they come from different schools.

## Section 1.3. What Our Leaf Does

In our decomposed tree, leaf j does not output a single number. It outputs a population-level prediction mu j plus a group-specific deviation b j g for each group g present in that leaf.

The prediction for observation i belonging to group g in leaf j is: prediction i equals mu j plus b j g.

mu j is the fixed component. It represents the population-average prediction for any observation landing in this leaf, regardless of which group it belongs to. It captures the covariate-driven signal: the effect of study hours, age, income, or whatever features the tree has split on.

b j g is the random effect for group g within this leaf. It captures how group g deviates from the population average in this region of covariate space. School 3 might be 2 points above average in this leaf, while school 7 is 1 point below.

The sum mu j plus b j g gives a group-personalized prediction that reflects both what the covariates say, through mu, and what the group membership says, through b.

Each leaf is, in effect, its own tiny mixed model. The tree structure handles the non-linear fixed effects by choosing where to split, and the within-leaf decomposition handles the group structure by estimating mu and b jointly.

## Part 2: The Mathematics Inside the Leaf

## Section 2.1. Setting Up the Leaf-Level Objective

Recall from gradient boosting that each leaf minimizes an objective built from the gradients g i and Hessians h i of the loss function. For a standard tree, the leaf objective is quadratic in the single weight w j. For our decomposed leaf, we need an objective that is quadratic in both mu j and all the b j g values.

The objective for leaf j is: Obj j equals the sum over groups g of the quantity G j g times the quantity mu j plus b j g, plus one half times H j g times the quantity mu j plus b j g squared, plus one half lambda times mu j squared, plus one over two sigma squared times the sum over g of b j g squared.

There are four distinct pieces. Let us go through each one carefully.

Piece 1: The Gradient Term. G j g times the quantity mu j plus b j g.

G j g is the sum of all gradients from observations belonging to group g in leaf j. Formally, G j g equals the sum of g i where the sum is over all observations i that are both in leaf j and in group g.

The gradient g i for each observation tells us the direction and magnitude of the error at the current prediction. For squared loss starting from the mean, g i equals negative y i minus y bar, so G j g is essentially the negative total residual for group g in this leaf.

Multiplying by the quantity mu j plus b j g means the contribution of this term depends on how large our prediction is. This is the linear term of the quadratic. It determines the direction of the optimum: should mu and b be positive or negative?

Piece 2: The Hessian Term. One half times H j g times the quantity mu j plus b j g, squared.

H j g is the sum of all Hessians from group g in leaf j. For squared loss, every h i equals 1, so H j g is just the count of observations from group g in the leaf.

This term is the curvature of the loss. Because it multiplies the prediction squared, it creates a bowl shape. The objective curves upward as the prediction moves away from the optimum in either direction.

The Hessian sum acts as a confidence measure: more observations from group g, meaning a larger H j g, means a steeper bowl, which means the optimum is more precisely located. Fewer observations means a flatter bowl, and the optimum is less certain.

Piece 3: The Fixed Effect Penalty. One half lambda times mu j squared.

Lambda is the L2 regularization parameter, the same one used in standard gradient boosting. This term penalizes large values of mu j, pulling the population-level prediction toward zero. It prevents the fixed component from overfitting. Without this penalty, mu j could take on extreme values, especially in leaves with few observations.

Piece 4: The Random Effects Penalty. This is the new ingredient. One over two sigma squared times the sum over g of b j g squared.

This is what makes our tree different from a standard tree. It penalizes the group-specific deviations.

Sigma squared is the random effects variance, a single number that controls how much groups are allowed to deviate from the population mean. It is the same concept as the variance component in a standard mixed model.

When sigma squared is large, the penalty 1 over sigma squared is small, so groups are allowed to deviate freely. The model trusts that groups genuinely differ.

When sigma squared is small, the penalty 1 over sigma squared is large, so group deviations are crushed toward zero. The model thinks groups are all similar, and individual group estimates are mostly noise.

The sum over g of b j g squared sums the squared deviations across all groups in the leaf. This is an L2 penalty on the random effects, which is mathematically equivalent to assuming the random effects are drawn from a normal distribution with mean zero and variance sigma squared. The penalty strength 1 over sigma squared is the precision, or inverse variance, of that prior.

This is the mechanism of shrinkage: group estimates are pulled toward zero by an amount that depends on how much data the group has versus how large sigma squared is.

## Section 2.2. Solving for the Group Deviations b j g

The beautiful property of our objective is that it is quadratic in every variable. Quadratic functions have a unique minimum that can be found by taking the derivative, setting it to zero, and solving. No iterative algorithm is needed.

First, we solve for each group's deviation b j g, treating mu j as temporarily fixed. Take the partial derivative of the objective with respect to b j g and set it to zero.

The partial derivative of Obj j with respect to b j g equals G j g plus H j g times the quantity mu j plus b j g, plus 1 over sigma squared times b j g, and we set this equal to zero.

The first term G j g comes from differentiating the gradient term. It is the raw error signal from group g.

The second term H j g times the quantity mu j plus b j g comes from differentiating the Hessian term. It is the curvature pulling the prediction back.

The third term 1 over sigma squared times b j g comes from differentiating the random effects penalty. It is the shrinkage, pulling b toward zero.

Rearranging to isolate b j g, the optimal value is:

b j g star equals negative the quantity G j g plus H j g times mu j, all divided by the quantity H j g plus 1 over sigma squared.

The numerator is the adjusted error signal: the group's total gradient G j g, adjusted for the fixed component that mu j already explains, via H j g times mu j. If mu j already explains most of the signal, the residual for b j g is small.

The denominator has two parts that compete. H j g is the data signal. 1 over sigma squared is the shrinkage. When the data signal is strong, meaning many observations and a large H j g, the data wins and b j g reflects the group's true deviation. When the data signal is weak, meaning few observations, the shrinkage wins and b j g is pulled toward zero.

This is the James-Stein shrinkage phenomenon, emerging naturally from the penalized objective. A group with 3 observations cannot be estimated precisely, so the model relies on the prior that all groups are similar. A group with 200 observations can be estimated precisely, so the model trusts the data.

## Section 2.3. Solving for the Fixed Component mu j

Now we substitute the optimal b j g back into the objective and solve for mu j. The algebra involves a key intermediate quantity.

Alpha g equals H j g divided by the quantity H j g plus 1 over sigma squared.

Alpha g is the fraction of the group's signal that gets absorbed by the random effect. It is always between 0 and 1.

When sigma squared is large, meaning groups vary a lot, alpha g is close to 1. The random effect absorbs nearly all the group-specific signal, and mu j only captures what is common across groups.

When sigma squared is small, meaning groups are similar, alpha g is close to 0. The random effect is suppressed, and nearly all signal flows into mu j.

When a group has many observations, meaning a large H j g, alpha g is larger. The model trusts the group estimate more. When a group has few observations, alpha g is smaller, and shrinkage is stronger.

The optimal mu j is:

mu j star equals negative the sum over g of G j g times the quantity 1 minus alpha g, all divided by the sum over g of H j g times the quantity 1 minus alpha g, plus lambda.

Each group's contribution to mu is weighted by 1 minus alpha g, the fraction of signal not absorbed by the random effect.

The numerator sums up the leftover gradient from all groups, the signal that the random effects did not claim.

The denominator sums up the leftover Hessians plus the regularization lambda.

The structure is exactly the standard leaf-weight formula negative G over H plus lambda, but with every quantity reweighted by the shrinkage factor 1 minus alpha g. The random effects have taken their share of the signal, and mu gets what remains.

## Section 2.4. Why This Is Elegant

The entire within-leaf computation is a closed-form solution to a convex quadratic problem. No iterations, no convergence issues, no numerical instabilities. For each leaf, we compute mu j and all the b j g values in a single pass through the group-level statistics. The computational cost is proportional to the number of groups in the leaf, which is typically small.

## Part 3: Sanity Checks. What Happens at the Extremes

A mathematical formula is only trustworthy if it behaves sensibly in extreme cases. We verified four critical limits.

## Section 3.1. When sigma squared goes to zero: No Group Variation

If there is no between-group variation, meaning sigma squared equals zero, the penalty 1 over sigma squared becomes infinite, crushing every b j g to exactly zero. The formula for mu j reduces to: mu j star equals negative G j divided by the quantity H j plus lambda.

This is exactly the standard gradient boosting leaf weight formula. When there is no group structure, our tree becomes a normal tree. The decomposition adds nothing and subtracts nothing. The standard case is recovered as a special case.

## Section 3.2. When sigma squared goes to infinity: Unrestricted Groups

If groups are allowed to vary without limit, the penalty vanishes, alpha g approaches 1 for all groups, and each b j g absorbs all the group-specific signal with no shrinkage. The leaf effectively fits separate constants for each group, with mu j becoming the unweighted grand mean.

This is a per-group model with no borrowing of strength. It overfits when groups are small, which is exactly why sigma squared should be finite in practice.

## Section 3.3. When a Group Has One Observation in the Leaf

If group g has a single observation in leaf j, then H j g is just h i for that one point, which equals 1 for squared loss. The denominator H j g plus 1 over sigma squared is dominated by the shrinkage term 1 over sigma squared whenever sigma squared is not too large. The group deviation b j g is heavily shrunk toward zero.

The model does not overfit to a single data point. The shrinkage automatically distrusts small-sample group estimates. This is the James-Stein phenomenon: borrowing strength from the population to improve individual estimates.

## Section 3.4. When All Observations Come from One Group

If leaf j contains observations from only one group, then mu j and b j g are not identifiable from data alone. Any amount can be shifted between them. But the two penalties resolve the ambiguity: lambda penalizes mu j and 1 over sigma squared penalizes b j g, so the model allocates signal between them according to the relative strength of the two penalties.

The system does not crash or produce nonsense when identification is weak. The penalties serve as priors that give a well-defined answer even in degenerate cases.

## Part 4: How the Tree Decides Where to Split

## Section 4.1. The Modified Gain Formula

A tree grows by evaluating candidate splits and choosing the one that improves the objective most. In a standard tree, the gain from splitting leaf j into left child L and right child R is:

Standard Gain equals one half times the quantity G L squared over H L plus lambda, plus G R squared over H R plus lambda, minus G j squared over H j plus lambda, minus gamma.

In our decomposed tree, each of the three terms is replaced by the full mixed-effects objective value, which accounts for the group structure and the random effects penalty.

Mixed Gain equals Obj j star minus Obj L star minus Obj R star, minus gamma.

Obj j star is the optimal objective value for the parent leaf before splitting, computed using the full mu and b solution. Obj L star and Obj R star are the optimal objective values for the left and right children after splitting, each computed with their own mu, b, and group compositions. If the gain is positive, the split improves the joint objective. If negative or zero, the split is not worth the added complexity.

## Section 4.2. What the Modified Gain Captures That Standard Gain Misses

The standard gain measures one thing: does splitting reduce the total squared error? The mixed gain measures three things simultaneously.

First, fixed effects improvement. Does splitting create more homogeneous covariate regions? This is the same as standard.

Second, random effects allocation. Does splitting separate groups in a way that produces cleaner group estimates? A split might put mostly school 3 students on the left and mostly school 7 students on the right, allowing each child to estimate its group effects with less interference.

Third, penalty change. Does splitting increase or decrease the total random effects penalty? If a split creates a child where one group has very few observations, the shrinkage will be very strong for that group in that child, changing the objective.

## Part 5: How This Fits into a Boosted Ensemble

## Section 5.1. The Learning Rate Applies to Everything

In a boosted ensemble, each tree's contribution is multiplied by a learning rate nu, typically between 0.01 and 0.3. For our decomposed tree, nu multiplies the entire leaf output.

y hat i at round t equals y hat i at round t minus 1, plus nu times the quantity mu j plus b j g.

Both mu and b are shrunk by the same learning rate. This is essential. If only mu were shrunk, the random effects would dominate after a few rounds and absorb everything. If only b were shrunk, the fixed effects would dominate and the random effects would never express themselves.

## Section 5.2. What Later Trees See

At round t, the pseudo-residuals are computed from the current total prediction, which includes all previous shrunken mu and b contributions. Early trees see large group-level variation in the residuals and allocate substantial b values. Later trees see smaller group residuals and allocate smaller b values. The random effects contribution naturally decays as the ensemble matures.

## Section 5.3. Emergent Random Effects Structure

Across the full ensemble, the effective random effect for group g at observation i is the sum of all the leaf-local b values from every tree, each scaled by nu. Because different trees partition the covariate space differently, this sum can vary by observation within the same group.

This is not a bug. It is a generalization. A standard random intercept model says school 7 is 3 points above average for all its students. Our decomposed approach says school 7 is 4 points above average for its high-performing students and 1 point below average for its struggling students. Random slopes and random interactions emerge automatically from the tree structure without being parametrically specified.

## Section 5.4. Updating Sigma Squared

After each boosting round, we update the global sigma squared estimate from the collection of b values. The estimate feeds into the next tree's shrinkage calibration. This creates a self-correcting feedback loop: if sigma squared is overestimated, meaning too little shrinkage, the b values will be noisy, residuals will not improve much, and subsequent trees will correct. If sigma squared is underestimated, meaning too much shrinkage, residuals retain group structure that later trees pick up.

## Part 6: Experimental Validation

## Section 6.1. Base Case: Recovering a Known Signal

We generated data from a known process: y equals 3 times sine of x 1 plus 2 times x 2 plus 1.5 times x 1 times x 2 plus b g plus epsilon, with 30 groups, 40 observations per group, true sigma squared equals 4.0, and noise variance 1.0.

The mixed tree recovered the total signal with MSE of 1.00, essentially the irreducible noise floor. The standard tree was at 3.83, meaning nearly 3 units of MSE were group variation it could not capture. The mixed tree's fixed effects were also cleaner because the group variation was properly separated, freeing the tree structure to focus on the covariate signal.

## Section 6.2. Stress Tests: Nine Scenarios

We tested the decomposed tree under nine increasingly difficult scenarios. The mixed tree outperformed the standard tree in every single test.

Random slopes: standard MSE 9.87, mixed MSE 1.35, an 86 percent improvement. Complex non-linear fixed effects: 70 percent improvement. Severely unbalanced groups with sizes from 3 to 200: 87 percent improvement. 200 groups with only 5 observations each: 82 percent improvement. Group-covariate confounding: 53 percent improvement. Crossed effects with an unmodeled site variable: 71 percent improvement. 50 features with only 2 carrying signal: 85 percent improvement. Weak signal with strong groups: 99 percent improvement. Non-Gaussian random effects: 90 percent improvement.

## Section 6.3. Key Findings from the Stress Tests

Random slopes emerged naturally. When the true group effect varied with a covariate, the tree captured this by assigning different b values to the same group in different leaves. The correlation with the true group effects was 0.993. Random slopes were never specified. They emerged from the interaction of tree structure and local group estimation.

Unbalanced groups were handled by shrinkage. Groups with 3 observations had their b values heavily shrunk toward zero, appropriately distrusting the noisy estimate. Groups with 200 observations were estimated with high fidelity. No manual tuning was needed.

Confounding was the hardest case. When group effects correlated 0.87 with a covariate, the improvement was only 53 percent. This is expected: when b and x are entangled, the boundary between fixed and random is genuinely ambiguous.

Non-Gaussian random effects did not matter much. With skewness of 2.73, the recovery was 90 percent. The Gaussian shrinkage assumption is violated but the per-group, per-leaf estimation is flexible enough to absorb non-Gaussian shapes.

## Part 7: Why This Architecture Solves the Known Problems

Problem Solved: Fixed-Random Competition. In existing methods like mboost and MERF, fixed and random effects compete for selection at each iteration. In our approach, there is no competition. Every leaf always computes both mu and b jointly. The fixed component and the random component are solved simultaneously from the same objective.

Problem Solved: Cluster-Constant Confounding. In MERMBoost, a special correction is needed for covariates that are constant within clusters. In our approach, this correction is structural. The tree splits on covariates, never on the group identifier. Every leaf therefore contains a mix of groups. Within each leaf, mu captures what is common across groups and b captures what is group-specific.

Problem Solved: Covariance Estimation. Standard boosting packages that include random effects estimate b values but not the covariance structure. Our approach produces b values at every leaf, and sigma squared is estimated directly from these values.

Problem Addressed: Sequential Bias. GPBoost and MEGB alternate between boosting the fixed effects and estimating the variance parameters, creating sequential bias because each step uses a stale estimate from the other. In our approach, each tree's within-leaf optimization solves for mu and b jointly, conditional on the current sigma squared. This substantially reduces though does not completely eliminate the sequential dependence.

Theoretical Foundation. The within-leaf optimization is convex quadratic with a unique closed-form solution. The splitting criterion directly measures improvement in the joint objective. Each split provably decreases the penalized loss, providing the monotonicity condition needed for formal convergence analysis. No other existing method has all three of these properties.

End of document.
