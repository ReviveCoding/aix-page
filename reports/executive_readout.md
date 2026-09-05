# Recommendation: ITERATE

**Hypothesis.** Richer and contextual search-ad experiences can raise advertiser value without degrading whole-page outcomes.

**Population.** Predictive context comes from 149.6M public KDD aggregate rows. Product evidence comes from a separate 17.0M-session controlled semi-synthetic randomized experiment.

**Primary result.** The frozen rich-product/top/one-ad finalist increased simulated advertiser value 40.90% relative to control (absolute +0.007125; 95% CI 0.006513–0.007738).

**Uncertainty and trust.** User-level SRM passed (p=0.9281); A/A, power, assignment, balance, missingness, and replay gates passed.

**Guardrails and segments.** All frozen finalist guardrails passed, and primary lift was positive in all six pseudo-weeks. Segment evidence is strongest in commercial and high-relevance contexts; unsupported exploratory subgroup claims are omitted.

**Risk.** The contextual P6 policy was safe but delivered 3.93% less DR advertiser value than the strongest safe static policy on the locked policy holdout.

**Decision.** Do not launch P6. Retain the safe static candidate as the benchmark and iterate policy modeling under the existing hard-constraint framework.

**Next experiment.** In a real product environment, validate the static treatment first under an independently governed protocol, then test a revised contextual policy against that static baseline with pre-registered user and page guardrails.
