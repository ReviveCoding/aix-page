# AIX-Page technical report

## Scope

AIX-Page is a portfolio research benchmark for whole-page Search Ads decisions. It deliberately separates **real public data**—KDD Cup 2012 Track 2 modeling and context—from **controlled semi-synthetic data**—randomized product-treatment outcomes. This is not production evidence and no result is a Google claim.

## Data and ownership

The official labeled KDD source contributed 149,639,105 aggregate rows, 235,582,879 impressions, and 8,217,633 clicks. A strict 12-field streamed parser validates binomial consistency and writes partitioned Parquet; DuckDB builds sponsored-search facts, dimensions, feature/context marts, and reporting marts. Raw archives stay immutable and are not redistributed.

Stable setting-key hashes prevent exact duplicate contexts crossing roles. The primary recurring-entity evaluation is accompanied by cold-query, cold-ad, cold-advertiser, anonymous-user, tail, support, depth, and position challenges. Target-dependent histories are five-fold cross-fitted in training and generated from training history only elsewhere.

## Predictive ladder

Rows use grouped-binomial likelihood with impressions applied once as trial weight. Finite-difference and grouped-versus-expanded tests qualify the custom objective and weighted metrics. M0 prior, M1 hashed linear, M2 XGBoost CUDA, and M3 DCNv2 CUDA were evaluated. M2 won validation; isotonic calibration won on a separate calibration role; the labeled locked test opened once. Deep learning did not win and was not promoted.

## Experiment qualification and design

The product DGP, oracle firewall, analysis plan, randomization, margins, sample size, and release thresholds were frozen before outcomes. An independent 240,000-row qualification pilot estimated advertiser-value SD 0.24518. With multiplicity-aware alpha and 85% target power, advertiser value required 1,414,820 sessions per arm; 12 arms produced 16,977,840 sessions.

Five hundred A/A replications produced 5.6% Type-I error (Monte Carlo interval 3.58%–7.62%) and 94.4% coverage (92.38%–96.42%). Treatment assignment uses stable SHA-256 at `sim_user_id`, and 20 jackknife buckets use a distinct stable namespace. Locked Tier-0 checks include randomization-unit SRM, assignment consistency, contamination, missingness, causal ordering, balance, traffic, and telemetry.

## Inference

The primary estimand is ITT advertiser value per eligible session. Reports include absolute/relative effects, stable-user jackknife standard errors and intervals, practical significance, and Holm-adjusted confirmatory p-values. User utility and organic engagement use non-inferiority lower bounds; abandonment, low-relevance exposure, and density use upper safety margins. CUPED uses pre-treatment covariates. ANCOVA and user/query cluster bootstraps assess sensitivity. The factorial model includes format, position, density, format×position, and format×density. Exploratory multiplicity uses correct BH-FDR.

Pre-specified HTE covers commercial/informational intent, relevance, head/tail queries, cold/returning simulated users, and prior exposure. Discovery and validation are separated; no empty CATE table is inflated into a claim. Six pseudo-weeks and Treatment×Week analysis evaluate the already-frozen finalist rather than choosing one after outcomes.

## Policy and delayed oracle

Policy roles split persistent users 60/20/20. Discovery fits outcome/CATE models; selection freezes architecture, thresholds, P6, and the strongest safe non-AIX comparator; locked test opens once. Known propensities support IPS, SNIPS, and primary DR OPE with clustered uncertainty. Safety is evaluated for each policy and hard constraints are never converted into an arbitrary score.

Oracle code is import-isolated until policy evaluation is immutable. Delayed diagnostics found mean absolute ATE bias 0.000196, CATE RMSE 0.007569, ranking Spearman 0.9909, and reported true values/regret; they did not permit reselection.

## Results and decision

The static rich-product/top/one-ad finalist delivered +0.007125 simulated advertiser value (+40.90%; 95% CI 0.006513–0.007738), passed its frozen guardrails, and was positive in all six pseudo-weeks. P6 was safe but had locked DR value 0.017976 versus 0.018710 for P1, a -3.93% relative difference. The decision hierarchy therefore returns **ITERATE**.

## Scientific governance

Two invalid protocols remain permanently unclaimable and a third was aborted before outcome generation. The valid protocol `KDDLOCK-58218173ab2542cb` replay-verified every frozen input and outcome chunk. Final QA reports 31 tests, 93% package coverage, 100% oracle-evaluator coverage, strict mypy and Ruff passes, package build, clean-wheel import, Windows smoke, and frozen replay passes.

## Limitations

KDD data are historical, hashed, and aggregated. The product DGP cannot establish real-market external validity, and simulated advertiser value is not revenue. Model qualification uses bounded role samples despite full-scale ingestion. Policy performance is conditional on the frozen DGP, action set, and constraints. A live product experiment would require new governance, telemetry, privacy review, and pre-registration.
