-- Canonical locked evidence marts are materialized by scripts/finalize_reporting_mart.py.
-- Paths are bound by that Windows-safe orchestrator; SQLite is not used.
CREATE OR REPLACE VIEW mart_experiment_outcomes AS
SELECT * FROM read_parquet($outcome_glob);

CREATE OR REPLACE TABLE mart_policy_evaluation AS
SELECT * FROM read_csv_auto($policy_values_csv, header=true);

CREATE OR REPLACE TABLE mart_primary_reporting AS
SELECT * FROM read_csv_auto($primary_effects_csv, header=true);
