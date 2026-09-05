# Methodology

## Real public data

KDD Cup 2012 Track 2 contains aggregated sponsored-search rows. Each row is modeled as `Click` successes among `Impression` binomial trials. The strict parser enforces all 12 fields, `Impression > 0`, and `0 <= Click <= Impression`. The public evidence reports aggregates only.

Stable setting-group hashes allocate 50%/10%/10%/10%/15%/5% roles for training, validation, calibration, locked test, experiment context, and challenges without duplicate-setting crossings. Five-fold OOF smoothed response histories prevent target leakage. Explicit cold-query, cold-ad, cold-advertiser, anonymous-user, tail, support, depth, and position challenges supplement recurring-entity evaluation.

For raw margin `z`, `p = sigmoid(z)`, response `y = Click / Impression`, and weight `w = Impression`, the grouped-binomial derivatives are `g = w(p-y)` and `h = wp(1-p)`. Tests cover finite differences, grouped/expanded likelihood and optimization direction, and weighted AUC equivalence. The ladder compares an empirical prior, hashed linear model, XGBoost CUDA, and DCNv2 CUDA. Calibration uses a separate role; the labeled locked test opens once.

## Controlled semi-synthetic experiment

The experiment is not KDD traffic and not production evidence. A frozen DGP generates advertiser value, user utility, abandonment, reformulation, organic engagement, low-relevance exposure, and density outcomes across a 3×2×2 factorial. Persistent `sim_user_id` values receive stable 12-arm assignment. Six pseudo-weeks provide temporal evidence.

Pre-lock gates include 500 A/A replications, pilot-derived clustered variance, multiplicity-aware power, oracle-firewall tests, and an independent readiness audit. The locked run contains 16,977,840 sessions. SRM is computed on 5,377,461 randomized users—not sessions—and precedes analysis.

The primary estimand is ITT advertiser value per eligible session. Primary uncertainty is a 20-bucket stable-user jackknife. CUPED uses pre-treatment covariates; ANCOVA and user/query cluster bootstraps are sensitivities. Confirmatory families use Holm adjustment; exploratory HTE uses BH-FDR. Factorial inference estimates format, position, density, format×position, and format×density rather than substituting cell means.

## HTE and policies

Commercial/informational, relevance, head/tail, cold/returning, and exposure segments were pre-specified. No separate exploratory CATE superiority claim is made where the canonical validation table is empty.

Policy discovery, selection, and locked test are disjoint at the user level. Known randomization propensities permit IPS, SNIPS, and doubly robust OPE. User/page constraints remain hard constraints and are estimated for every policy. P6 was frozen before its one-time policy holdout; delayed oracle analysis could evaluate error, coverage, ranking, true value, regret, and violations but could not trigger reselection.

## Governance history and decision

One synthetic qualification protocol was invalidated for assumed variance, outcome-before-identity ordering, and missing policy guardrail OPE. A KDD protocol was invalidated because SRM used sessions rather than randomized users. A replacement identity was aborted before outcomes after a seed-governance issue. Compact summaries are published; their effects are not claimable.

The valid protocol is `KDDLOCK-58218173ab2542cb`. The static finalist passed its locked practical and guardrail gates, but the safe P6 contextual policy underperformed the strongest safe static baseline by 3.93%. The frozen hierarchy therefore yields **ITERATE**.
