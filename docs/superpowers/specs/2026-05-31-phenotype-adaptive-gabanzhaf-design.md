# Probabilistic Causal Banzhaf Design

## Goal

Improve `RCAEval/e2e/gabanzhaf.py` with a simpler and more defensible RCA algorithm that can be written as one coherent academic method.

The design target is to push `AC@1` and `Avg@5` toward `0.9+` across all fault types while preserving these constraints:

- fully unsupervised;
- no ground truth, fault label, dataset name, case ID, or service-name exception;
- no fallback branch;
- one unified flow for every incident;
- small runtime increase after candidate pruning;
- fewer ad hoc helper stages than the current phenotype-adaptive draft.

The second iteration changes the core cooperative game instead of stacking more post-processing terms.

## Motivation

The first phenotype-adaptive implementation added useful evidence, but the algorithm became harder to justify:

```text
 metric anomaly
+ trace evidence
+ semantic evidence
+ phenotype inference
+ residual victim suppression
+ topology weighting
+ metric ranking correction
```

This creates a weak paper story because every helper needs a separate theoretical justification.

The observed benchmark failures also suggest that the main problem is not missing telemetry evidence. The main problem is the current Banzhaf payoff:

```text
v(S) = sum_j W_j * I[j is explained by S]
```

This binary coverage payoff over-rewards shared dependencies such as common databases, caches, and message brokers. If a shared dependency can topologically explain many anomalous services, it receives excessive marginal contribution even when it is only a propagated victim or a common downstream component.

The improved design therefore modifies only the cooperative game:

```text
old: binary causal coverage game
new: probabilistic causal coverage game
```

The evidence extraction and candidate pruning remain lightweight. The final ranking becomes a projection of the Banzhaf score back to metrics, without extra fault-type correction.

## Core Contribution

The proposed method is:

```text
Probabilistic Causal Banzhaf
```

Its contribution is a single mathematical replacement for the current binary coverage payoff:

1. Model root-cause explanation as a probabilistic causal coverage game.
2. Use a noisy-OR coverage function to encode diminishing marginal returns when multiple services explain the same anomaly.
3. Compute exact Banzhaf power on a fixed top-K unsupervised candidate game.
4. Use one flow for every incident, with no fallback and no label-specific branch.

This is theoretically grounded in:

- Banzhaf power index for expected marginal contribution in cooperative games;
- noisy-OR probabilistic coverage from probabilistic graphical models;
- submodular coverage / influence models, where repeated explanation has diminishing returns.

## Unified Pipeline

The algorithm flow is fixed:

```text
1. Split pre/post windows.
2. Compute metric anomaly evidence.
3. Aggregate metric evidence to service anomaly weights.
4. Build bounded causal topology from lag correlation and trace call relations.
5. Select a fixed top-K unsupervised candidate set.
6. Build causal explanation probability matrix P.
7. Compute exact Banzhaf on the top-K probabilistic coverage game.
8. Project service Banzhaf scores back to metric ranking.
```

There is no runtime branch such as:

```text
if K is large: use deterministic Banzhaf
if delay: boost latency
if loss: boost errors
if dataset is re2: change behavior
```

The method always uses exact Banzhaf on the same truncated candidate game.

## Notation

Let:

- `V` be the bounded set of observed services;
- `M` be the observed metrics;
- `P_K subset V` be the top-K candidate service set;
- `i in P_K` be a candidate root service;
- `j in V` be a service whose anomaly may be explained;
- `m in M` be a metric;
- `A_j in [0, 1]` be service-level anomaly weight;
- `E_m in [0, 1]` be metric-level anomaly evidence;
- `T_ij in [0, 1]` be topology strength from service `i` to service `j`;
- `L_ij in [0, 1]` be temporal precedence or lag-correlation strength;
- `S_ij in [0, 1]` be anomaly-alignment strength between services `i` and `j`;
- `p_ij in [0, 1]` be the probability that candidate `i` causally explains anomaly at service `j`.

All quantities are computed from telemetry only.

## Candidate Set

The exact game is intentionally computed on a fixed-size unsupervised candidate set:

```text
P_K = TopK_i phi_i
```

where:

```text
phi_i =
    w_a * A_i
  + w_r * R_i
  + w_t * Trace_i
  + w_g * G_i
```

`R_i` is residual local anomaly after upstream explanation, `Trace_i` is trace anomaly if available, and `G_i` is bounded graph participation evidence.

This is not a fallback. The truncated game is the algorithm definition. Every incident uses exactly the same candidate selection rule and exactly the same `K`.

Recommended default:

```text
K = 12
```

This gives `2^12 = 4096` coalitions per case, which is small after service-level pruning.

## Causal Explanation Probability

The explanation probability matrix replaces binary explainability.

For candidate `i` and service `j`:

```text
p_ij =
    1,                                  if i = j
    sigma(T_ij * L_ij * S_ij),          otherwise
```

where `sigma` is a bounded calibration function:

```text
sigma(x) = 1 - exp(-lambda * x)
```

`T_ij` comes from the existing bounded topology matrix. `L_ij` comes from lag correlation or trace temporal direction. `S_ij` is anomaly alignment:

```text
S_ij = sqrt(A_i * A_j)
```

This keeps high explanation probability only when three conditions agree:

1. `i` is structurally able to influence `j`;
2. the temporal relation is plausible;
3. both services are anomalous.

Shared dependencies are no longer automatically rewarded for every connected anomaly. They must provide high marginal coverage under the noisy-OR game.

## Noisy-OR Coverage Payoff

For a coalition `S subseteq P_K`, service `j` is probabilistically covered by:

```text
c_j(S) = 1 - prod_{i in S} (1 - p_ij)
```

The game payoff is:

```text
v(S) = sum_{j in V} A_j * c_j(S)
```

This is the central simplification.

The old binary payoff gives no diminishing return once a service is explainable. The new payoff is submodular: if one candidate already explains `j`, adding another candidate that explains the same `j` contributes less. This directly addresses shared-dependency over-ranking without an explicit shared-dependency penalty.

The marginal contribution of candidate `i` is:

```text
Delta_i(S)
= v(S union {i}) - v(S)
= sum_j A_j * p_ij * prod_{h in S} (1 - p_hj)
```

This formula shows the desired behavior:

- if no existing coalition member explains `j`, `i` receives high marginal contribution;
- if many services already explain `j`, `i` receives low marginal contribution;
- duplicated explanations are naturally discounted.

## Exact Banzhaf Score

The Banzhaf score for candidate `i` is:

```text
beta_i =
1 / 2^(K-1)
sum_{S subseteq P_K \ {i}} Delta_i(S)
```

Using the noisy-OR marginal form:

```text
beta_i =
1 / 2^(K-1)
sum_{S subseteq P_K \ {i}}
sum_j A_j * p_ij * prod_{h in S} (1 - p_hj)
```

This is exact Banzhaf on the truncated candidate game. It is not Monte Carlo and not deterministic approximation.

## Closed-Form Optimization

The noisy-OR Banzhaf score has a closed-form simplification.

For fixed `i` and `j`:

```text
1 / 2^(K-1)
sum_{S subseteq P_K \ {i}} prod_{h in S} (1 - p_hj)
= prod_{h in P_K \ {i}} (1 + (1 - p_hj)) / 2
= prod_{h in P_K \ {i}} (1 - p_hj / 2)
```

Therefore:

```text
beta_i =
sum_j A_j * p_ij * prod_{h in P_K \ {i}} (1 - p_hj / 2)
```

This is still exact Banzhaf for the noisy-OR game, but avoids enumerating all coalitions.

The implementation should use this closed form by default because it is mathematically equivalent and faster. The paper can describe it as an exact closed-form Banzhaf estimator for the noisy-OR causal coverage game.

## Metric Ranking

The final metric score should remain simple:

```text
score(m) =
    beta_service(m) * (epsilon + E_m)
```

where:

- `beta_service(m)` is the normalized Banzhaf score of the service that owns metric `m`;
- `E_m` is the local metric anomaly evidence;
- `epsilon` preserves service-level root evidence when the root metric is not the single largest shifted metric.

No extra semantic or fault-type ranking term is added.

This keeps the paper story clean:

```text
service root probability comes from Banzhaf;
metric identity comes from local anomaly evidence.
```

## Expected Effect on Fault Types

The same mechanism applies across fault types:

- CPU, memory, disk, socket: self-explanation `p_ii = 1` plus high local anomaly should favor the true service over downstream victims.
- Delay: topology and temporal relation allow upstream causal explanation, but duplicated downstream coverage has diminishing return.
- Loss: trace/status anomalies contribute to service anomaly weights, while noisy-OR prevents common dependencies from dominating only because many services are affected.

This is not implemented as fault-specific logic. It follows from the same payoff.

## Configuration

Keep the configurable surface small:

```python
exact_candidate_k: int = 12
causal_probability_lambda: float = 1.5
metric_rank_epsilon: float = 0.15
```

Avoid introducing phenotype gains, semantic gains, dataset-specific knobs, or fallback thresholds.

## Complexity

Let:

- `T` be time-series length;
- `M` be metric count;
- `N` be bounded service count;
- `K` be the fixed candidate count, with `K << N`;
- `E` be bounded topology edge count.

Existing preprocessing remains:

```text
metric evidence      O(T * M)
service aggregation  O(M)
topology building    O(T * N^2)
metric ranking       O(M log M)
```

New probabilistic Banzhaf steps:

```text
candidate selection        O(N log K)
explanation matrix P       O(K * N)
closed-form Banzhaf        O(K * N)
```

Total:

```text
O(T * M + T * N^2 + M log M + K * N)
```

Because `K` is fixed, the new game layer is effectively linear in the bounded service count. It should not be the runtime bottleneck.

If the implementation includes an optional test-only enumerator to validate the closed form, its complexity is:

```text
O(2^K * K * N)
```

That enumerator must not be used in the production RCA path.

## Testing Strategy

Add focused tests for:

- noisy-OR coverage is monotone and bounded in `[0, 1]`;
- noisy-OR payoff has diminishing marginal returns;
- closed-form Banzhaf matches brute-force exact Banzhaf on a small synthetic game;
- shared duplicated explanations receive lower marginal contribution than unique local explanations;
- candidate truncation is deterministic and uses no labels;
- final metric ranking combines service Banzhaf and local metric evidence only;
- no dataset name, fault label, root-cause field, or service-name exception appears in the implementation.

Benchmark validation:

- run `gabanzhaf` on `re2-ob`, `re2-ss`, and `re2-tt`;
- compare `AC@1`, `AC@3`, `AC@5`, `Avg@5`, wall time, and average speed against the current result file;
- inspect any large regression, especially in CPU/SOCKET top-1 and DELAY/LOSS top-5;
- verify runtime remains close to the current budget.

## Non-Goals

- Do not add supervised training.
- Do not use ground-truth root cause or fault labels.
- Do not branch by dataset name, case ID, fault type, or service name.
- Do not add a separate semantic ranking correction.
- Do not add a phenotype-specific payoff branch.
- Do not add deterministic fallback or Monte Carlo fallback.
- Do not build a full metric-level causal graph.
- Do not introduce GNN, Hawkes training, or Neural Granger training.

## Implementation Scope

Primary file:

- `RCAEval/e2e/gabanzhaf.py`

Test file:

- `tests/test_gabanzhaf_improvements.py`

The implementation should reduce conceptual complexity where possible:

```text
window split
metric evidence
service anomaly aggregation
topology / trace relation
top-K candidate game
probabilistic explanation matrix
closed-form exact Banzhaf
metric ranking
```

Existing helper functions that only support phenotype or semantic correction should be removed or bypassed if they are no longer part of the method.
