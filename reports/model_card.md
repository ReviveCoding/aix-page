# Model card

M2 XGBoost CUDA was selected by frozen validation weighted log loss (0.133363), ahead of M1 hashed linear (0.141195), M0 prior (0.152995), and M3 DCNv2 CUDA (0.167030). Isotonic calibration was selected on the calibration role. The one-time 500,000-row labeled locked test produced weighted log loss 0.127129, weighted Brier 0.030351, AUC 0.7794, PR-AUC 0.1345, and ECE 0.000787.

The model estimates propensity to click from historical public data; it does not establish causal product effects. Hashed identifiers and historical age limit external validity.
