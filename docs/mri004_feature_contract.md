# MRI-004 — session features and forward targets

Status: implementation contract, 2026-11-16. No model, forecast, backtest, or real SPY result is included.

## Objective, hypothesis, and baseline

Engineering hypothesis: deterministic transforms of a validated SPY daily frame produce the specified features using only sessions through `t`, while target builders use exactly sessions `t+1` through `t+h`. Baseline is hand-calculated synthetic price paths and prefix-invariance checks, not a predictive financial model. Acceptance: exact window boundaries and units, first-valid indices, no future-row effect on prior features, targets aligned to their maturity session, rejection of invalid input, and tests passing offline.

Input is the canonical daily schema from `docs/data_contract.md`. Source adjustment basis must be checked in the source manifest before interpreting values; this module does not certify historical `available_at`. One row represents an XNYS session date; the adjusted close is a positive USD/share level. No resampling, interpolation, sorting, fitted transform, or random sampling occurs. The initial HMM's future input columns will be log return and 20-session realized volatility only. Momentum and drawdown remain ablation candidates. A close-based feature row `t` is at the earliest usable after close `t`, subject to actual provider publication evidence.

## Exact transforms

For session index `t` and adjusted close `P_t`:

- `log_return[t] = ln(P_t) - ln(P_(t-1))`, dimensionless; first row missing.
- `realized_volatility_20[t] = sqrt((252/20) * sum_(i=0..19) log_return[t-i]^2)`, annualized volatility; first valid at index 20, because 20 returns require 21 prices. This is uncentered RMS of **daily close-to-close** returns, not high-frequency realized variation or a fitted volatility estimator.
- `momentum_20[t] = ln(P_t) - ln(P_(t-20))`, dimensionless; first valid at index 20.
- `drawdown_60[t] = P_t / max(P_(t-59),...,P_t) - 1`, dimensionless, nonpositive; first valid at index 59. The current session belongs to the maximum window.

Missing warm-up values stay missing (`NaN`), never backfilled. Windows count rows of the already validated XNYS session table, not calendar days. The config `configs/features.yaml` fixes windows and annualization; seed is recorded for future stochastic work but has no role in these deterministic transforms.

Forward labels are **separate from the feature table**. For `h in {5,20}`, `forward_variance_h[t] = (1/h) * sum_(j=1..h) log_return[t+j]^2` in daily variance units; `forward_volatility_h[t] = sqrt(252 * forward_variance_h[t])` in annualized units. `label_end_session_h[t]` is the session date at `t+h`; the last `h` origins have missing target and end session. The feature builder never receives targets. Any fit at time `T` may use only labels with `label_end_session_h <= T` and must purge labels crossing validation/final boundaries; MRI-006 will enforce this. Event labels require a training-window threshold and are deliberately deferred to evaluation code.

## Hand-checkable examples

For prices `[100, 110, 121]`, log returns at indices 1 and 2 both equal `ln(1.1)`. A two-return synthetic volatility with annualization 252 equals `sqrt(252) * ln(1.1)` at index 2. Two-session momentum at index 2 equals `ln(1.21)`. For `[100, 90, 95]`, a three-session drawdown at index 2 is `95/100 - 1 = -0.05`. For horizon 2 at origin 0 with `[100, 110, 121]`, forward variance is `ln(1.1)^2`, and its label ends at index 2.

## Source and limits

The equations above are the project's frozen local specification in `docs/research_protocol.md`, not an attributed empirical finding. Implementation semantics checked against [pandas 2.3.3 rolling documentation](https://pandas.pydata.org/pandas-docs/version/2.3.3/reference/api/pandas.Series.rolling.html) (`min_periods`, right-edge label) and [NumPy log documentation](https://numpy.org/doc/stable/reference/generated/numpy.log.html) (natural logarithm), consulted 2026-11-16. These are HTML technical references; no PDF hash or page number applies. No paper or book was used to infer predictive validity. Passing synthetic checks proves arithmetic and timing of this code, not source availability, point-in-time correctness of adjusted data, or risk-forecast performance.
