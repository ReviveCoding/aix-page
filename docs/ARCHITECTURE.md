# Architecture

![AIX-Page architecture](../reports/figures/08_architecture.png)

The real-data path is official KDD ZIP → strict streamed parsing → partitioned Parquet → DuckDB marts → stable entity-grouped roles → leakage-safe features → calibrated predictive model. The experimental path takes pre-treatment KDD-derived context into a separately frozen controlled semi-synthetic DGP, randomizes persistent simulated users, runs trust gates, estimates treatment effects and guardrails, evaluates constrained policies on a disjoint holdout, then opens delayed oracle diagnostics.

Raw data are immutable and stay local. Target-dependent historical features are out-of-fold. Model validation, calibration, locked test, experiment context, and challenge roles are isolated. Experiment assignment and jackknife buckets use deterministic cryptographic hashes, not process-randomized Python hashes.
