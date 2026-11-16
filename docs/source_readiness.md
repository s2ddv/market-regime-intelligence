# MRI SPY historical source readiness — 2026-11-16

Status: **candidate selected for sample audit; real dataset not accepted**. This document is a source-selection decision for MRI-003 follow-up, not an empirical result or authorization to run MRI-004–006.

## Requirement and evidence

The project needs one daily SPY series from 2005-01-01 to the latest completed XNYS session, explicit split/dividend adjustment, complete regular-session bars, stable raw bytes, a reproducible processing manifest, and terms permitting the user's own research. A current download cannot establish historical `available_at` or provider revisions. Source descriptions are claims to verify against a sample, not proof of data quality.

| Source | Primary-source evidence checked 2026-11-16 | Decision |
| --- | --- | --- |
| Stooq | Existing `stooq.com/q/d/l/` adapter returned HTTP 404 on a full-range request. Short direct probes with a browser user agent returned HTML verification pages, not CSV. Adjustment convention is undocumented in this project. | Keep the adapter for reproducibility; do not use as research input without a working CSV and adjustment evidence. |
| Yahoo Finance | [SPY history](https://finance.yahoo.com/quote/SPY/history/) labels Close as split-adjusted and Adj Close as split/dividend-adjusted. [Yahoo Help](https://help.yahoo.com/kb/finance/certain-amounts-sln2311.html) says offline CSV download requires Gold and may be unavailable for licensed instruments. A read-only probe of the chart JSON returned HTTP 200, but that endpoint is not a documented research-data contract here. | Do not ingest the undocumented JSON as the project's source. |
| Alpha Vantage | [Official API documentation](https://www.alphavantage.co/documentation/) documents raw daily OHLCV, while `TIME_SERIES_DAILY_ADJUSTED` provides adjusted close and actions and is marked Premium. | Suitable technical schema, but full adjusted history requires an entitlement/key not available in this workspace. |
| Marilux | [SPY page](https://marilux.dev/en/data/SPY) advertises daily history from 2003-09-10 through 2026-11-13, 5,833 trading days at review, daily download free, and split/dividend adjustment. Its [download page](https://marilux.dev/en/data?adj=qfq&f=day&from=2003-09-10&sess=ext&t=SPY&to=2026-11-13&tz=et) lists Daily as free and a CSV header `time,open,high,low,close,volume,vwap,trades`, with a Regular-only option. The [terms](https://marilux.dev/en/terms) permit one's own research and prohibit raw-data redistribution. | **Preferred candidate for a small authorized sample and full daily download**, subject to account access, exact file format and quality audit. Do not commit raw data. |

The [State Street SPY issuer page](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy) and [distribution history](https://www.ssga.com/us/en/individual/resources/documents/etf-dividend-distributions) are independent corporate-action references for checking a sample around ex-dividend dates. We have not downloaded or reconciled the distribution file yet.

## Controlled acquisition and acceptance gate

1. User obtains an authorized Marilux **Daily / SPY / 2005-01-01 to latest completed session / Split & dividend / Regular only / US Eastern** CSV in their own account. The data stay in ignored `data/raw/` or another local private path; never commit or redistribute them. No account, key, subscription, or contact with the vendor is initiated by this repository.
2. Inspect a short sample that includes ordinary sessions and at least one dividend ex-date before writing a Marilux adapter. Confirm exact header, date/time semantics, session definition (regular close versus extended-hours close), encoding, numeric units, ordering, action policy and any metadata supplied. Record URL/product settings, account entitlement without credentials, retrieval UTC, hash and terms version.
3. Compare expected XNYS sessions across the full requested interval; fail on gaps, duplicates, extras, invalid OHLC/volume or truncated coverage. Check one or more distribution dates and amounts against State Street. Compare price/adjustment behavior around ex-dates, including daily return continuity, without treating a single match as comprehensive verification. Audit requested end against the latest completed session.
4. Preserve exact source bytes and a separate content-addressed Parquet/manifest. Record `price_basis=adjusted_ohlc`, `adjustment_verified` only after the action checks, and source file settings. The source's adjusted historical values are revised to a current anchor; log-return ratios and relative drawdowns before a later action are invariant to a common multiplicative future factor, but revisions to prior factors and unavailable historical publication times remain unresolved.
5. Keep MRI-004 limited to synthetic fixtures until source coverage, adjustment, rights and timestamp gaps are resolved. A passed schema test is a software result, not permission to claim causal historical performance.

## Current blockers

No Marilux account or authorized CSV exists in this workspace. The actual Daily CSV format, full coverage, adjustment factors, historical revisions and publication timestamps are therefore **not demonstrated**. The Stooq live endpoint did not yield CSV. No real SPY snapshot was created or accepted in this step.
