# Publication audit

**PUBLICATION_AUDIT: PASS**

- Total files: 102
- Total bytes: 466,980
- Secret scan: PASS
- Raw/row-level data scan: PASS
- Absolute personal path scan: PASS
- Hard file-size gate (>90 MiB): PASS
- Warning file-size gate (>25 MiB): PASS
- Public provenance manifest: PASS
- README claim audit: PASS
- Technical-report claim audit: PASS

## Largest files

| File | Bytes | SHA-256 |
|---|---:|---|
| `reports/figures/05_policy_comparison.png` | 42,246 | `00761673271f9fa63f544802d9a9566444513e8d458fe1ec38e317c4401ddd24` |
| `reports/figures/03_guardrail_matrix.png` | 39,497 | `b036cb59ee23d79672d7eb67109d70454302422fb6e76c36ba7fa17d5ff1dc8d` |
| `reports/figures/02_treatment_forest.png` | 38,407 | `0de99492ebeabe8f4f49a2665b2ec0bd5b3dd69b8ad2cae0583d1237ae920a28` |
| `reports/figures/07_temporal_stability.png` | 35,342 | `31c41a0435bdfcb00b3eed4a1520a1871b258783a34798ceac7b14d40c1c9500` |
| `reports/figures/08_architecture.png` | 33,785 | `974f3783e3ff95db29b606016d531eb0f193fe12c95d87e047df29422310f824` |
| `reports/figures/01_pareto_frontier.png` | 26,077 | `7968d5f50bde4edd41c3c8382114a45e240751a9cf477256a1486b365d109862` |
| `reports/figures/04_hte.png` | 24,137 | `7108671d88874c6ea84a6cfdad4f8379dfd39da9aadcb93ffd6a164087371af4` |
| `reports/figures/06_calibration.png` | 16,778 | `159d0ccda131d43ec2753a51238ff26b38fe17cc4bd31f1ede6b896c19577473` |
| `scripts/train_real_models.py` | 16,617 | `2ff6ad7c0973d71973a0f849fa4d27b321a7d3c463dc31cd06687933d268a804` |
| `public_evidence/guardrail_effects.csv` | 12,314 | `a0e9f412f73a3490587f23680ebe1390e3b33690e3e61f2caeac9de67b0356ef` |
| `public_evidence/public_provenance_manifest.json` | 11,741 | `7001cc1560423e030cd717204e926a6c35da4cee08cf35b2b78469200b73d4fb` |
| `scripts/build_kdd_marts.py` | 9,452 | `4201a227edf504fc54e39aa76f73dda34eb1e80e6af594046c2c97bc5151a28e` |
| `scripts/build_kdd_features.py` | 8,914 | `6caca50a6b5dc8a841a6670d77bc1f5e46e780ef1862ab9c0db6310eef797b51` |
| `public_evidence/policy_values.csv` | 8,686 | `3bba55f0d6e2103070dc8e0c4bdfac2e60797f9173fd246b9c3eed15d7622613` |
| `src/aix_page/experimentation/inference.py` | 7,930 | `57051d0c4ec1a0eaf3cdd8bf7d9cd096bc578b6e62476f751b766b7197df4006` |
| `scripts/ingest_kdd.py` | 7,455 | `b1e48a3134212c6955661ee30f2f99fa4e934ea2c8c5350723ede780745839e8` |
| `README.md` | 6,706 | `41402c0f83b30460c7262aad8fe8fbacc0a59ff47c10db5bab20152973dd3a47` |
| `src/aix_page/simulator/observable_generator.py` | 6,527 | `cf4b5d32d38bde8387d81eafabdb6b24ffb4144ff244a025b85ada9277510c23` |
| `src/aix_page/policy/learning.py` | 5,357 | `041ac6e89ed3cae9a3964611cbd3a2ef334b34f4679c3d8a17e91e1ae6f4dc36` |
| `reports/technical_report.md` | 5,236 | `3fdfbb493042810152ccecb5f701b9d41c5497d1778d5ad273a5c03a3309a188` |

## Findings

No prohibited files, credentials, row-level data, private absolute paths, or oversized files were found.

The mirror is an allowlisted publication tree. KDD archives/rows, derived Parquet, DuckDB databases, model binaries, locked row-level outcomes, environments, and credential stores are excluded.
