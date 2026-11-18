# MRI-005 — Gaussian HMM and causal filtering

Status: implementation contract frozen before model code, 2026-11-16. This stage is limited to synthetic verification; no source has passed the SPY historical audit.

## Hypothesis, baseline, acceptance

Engineering hypothesis: a diagonal Gaussian HMM with 2–4 latent states can be fitted on a specified training prefix, and a forward filter using frozen parameters returns state probabilities at session `t` that are invariant to later observations. The baseline is an independently calculated small fixed-parameter filter and synthetic known-state sequences. Acceptance requires normalized finite probabilities, chronological input and training-only standardization, deterministic seeded fitting, no smoothing in the public filter, explicit training cutoff, and prefix-invariance tests. These checks do not establish financial predictive value or recovered economic regimes.

## Inputs and timing

Input columns are `date`, `log_return`, and `realized_volatility_20` from MRI-004; momentum, drawdown and all targets are excluded. Complete warm-up rows before the first finite volatility are omitted, while later missing or non-finite values are rejected. Each row is one validated XNYS session; gaps or revised adjusted prices must have been addressed by MRI-003 and the source audit. Fit uses only rows through `train_end`. Means and scales are computed on those rows and frozen. New rows must have dates strictly later than `train_end`; the filter starts from the posterior at that training end, advances one transition, then uses that row's likelihood. Thus its output at close `t` may be used only after close `t` and after actual source publication. `available_at` is not supplied by this model.

## Statistical specification

Validation clarification (2026-11-17): only the leading prefix with missing volatility is warm-up. Infinite inputs and negative volatility are rejected even in that prefix; after the first available volatility, both inputs must remain finite. The planner and model share this rule. Minimum state occupancy, finite likelihood and the existing likelihood-decrease tolerance are checked on the exact parameters returned by each start, including convergence and iteration-limit exits. A failed start cannot win selection; if every start fails, the error includes each start's diagnostic reason. Reaching the iteration limit still records `converged=false`; it does not waive the gates.

State `z_t` follows a homogeneous first-order Markov chain with initial probabilities `pi` and row-stochastic transition matrix `A`. Standardized two-dimensional observation `x_t` has one diagonal Gaussian density per state. Diagonal covariance is an explicit simplifying assumption, despite correlation between return and realized volatility. Multiple seeded starts and EM estimation are used; the best finite training log likelihood is retained. Variance floors and minimum occupancy protect numerical stability and are recorded with the model. Convergence is a numerical condition, not proof of global optimum. States are ordered by increasing fitted mean of realized volatility in original units, with a stable index tie-break; this label convention is local to each fit and does not prove cross-fit identity.

For fixed parameters, `p_t(j) ∝ b_j(x_t) sum_i p_(t-1)(i) A_(i,j)`. The public filter computes this forward recursion in log space. Backward probabilities are allowed **inside training EM only** to estimate parameters; they are not returned as historical real-time state estimates. A fitted model's parameters may depend on every observation in its training prefix, so training-period posterior values cannot be presented as out-of-sample causal forecasts. MRI-006 must enforce fit dates and retraining schedules before any evaluation.

## Literature consulted and limits

Lawrence R. Rabiner, *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*, Proceedings of the IEEE 77(2), 1989, pp. 257–286, DOI [10.1109/5.18626](https://doi.org/10.1109/5.18626). PDF copy from [Carnegie Mellon University](https://www.cs.cmu.edu/~durand/03-711/Readings/Rabiner89.pdf), SHA-256 `36f09aa771ed00201eaab1ffa93e5de01d2b85e577d07b34a4c8fe0918013d3a`. Consulted physical PDF pages 6–11, corresponding to printed pp. 262–267; equations 18–20 (forward recursion), 26–40 (smoothing and reestimation) and 49–54 (continuous Gaussian emissions). Equations on physical pp. 6 and 11 were visually checked. The paper's theoretical HMM treatment and speech applications support the algorithmic form, not a claim that HMM regimes improve SPY risk forecasting. The diagonal covariance, scaling, floors, seeds and validation gates are local design choices.

No real SPY data, backtest, variance forecast, event probability, GARCH/EWMA comparison or final holdout is used in this task. Full historical coverage, corporate-action adjustment, rights and point-in-time publication remain unverified.
