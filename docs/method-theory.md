# MERIT method definitions and closed-form derivation

This document is the source of truth for the paper's problem definition,
notation, and complexity claims. It describes the implementation jointly
defined by the seven `RCAEval/e2e/merit*.py` modules listed in
[`merit-ablation.md`](merit-ablation.md); it does not introduce a second algorithm.

## Problem and assumptions

For one diagnosed incident, the input is a multivariate metric time series
`X`, an anomaly-detection timestamp `t_a`, and, when available, distributed
trace records from the same system. The output is an ordered list of
service-metric indicators. Service-level evaluation removes the metric suffix
and duplicates from that list.

The retrospective evaluation assumes:

1. one injected root-cause service per case;
2. `t_a` is supplied by the benchmark (delay sensitivity perturbs it by the
   pre-registered `tdelta` values);
3. telemetry clocks are sufficiently aligned for pre/post and multi-lag
   comparisons;
4. metric and trace service names can be aligned by exact or data-driven prefix
   matching; and
5. each case uses only its own spans strictly before `t_a` to build the request
   graph; a case without usable pre-fault spans has an explicitly empty graph.
   Cache keys include dataset, case, boundary, and graph source, so graph state
   is never shared across cases.

MERIT is training-free: it does not fit a supervised or reusable predictive
model. It does have fixed unsupervised hyperparameters, shared by all three
systems.

## Evidence, directions, and candidate truncation

Metric evidence retains every metric row in the complete case record and splits
it at `t_a`. It robustly standardizes every post-`t_a` row against all available
pre-`t_a` rows using median/IQR statistics, then combines magnitude,
persistence, peak, and early/full-post behavior. Unequal pre/post lengths are
intentional. Metric scores are aggregated into a service anomaly weight
`w_j >= 0`.

All game edges use the direction "candidate cause explains affected target."
For a request edge `caller -> callee`, a fault in the callee can affect the
caller-visible latency, so dynamic propagation evidence is stored as
`callee -> caller`. Coverage rescue retains the request direction
`caller -> callee`: under its separately gated signature, a quiet caller that
has at least two loud latency callees is allowed to explain those callees.
Thus dynamic trace edges and coverage edges can point in opposite directions
for the same request pair. The call graph, dynamic propagation graph, and
coverage explanation edges must not be described as if they shared one edge
semantics.

Let `C` be the anomaly-ranked candidate pool and let `K_0` be its frozen base
cutoff. The game uses those base candidates plus any separately qualified
coverage parents, so the observed player count `K` can exceed `K_0`. Candidate
construction occurs before attribution and therefore sets an
upper bound on achievable service recall: if the true service is absent from
the game, no attribution rule can recover it. Candidate recall is computed only
by the offline evaluator and is reported separately, so a candidate-selection
gain is not credited to Banzhaf attribution.

## Directed coverage-strength game

Let `\mathcal P = {1, ..., K}` be candidate players and
`\mathcal T = {1, ..., N}` be target
services. For candidate `i` and target `j`, `a_ij in [0, 1]` is a bounded,
directed evidence strength saying how strongly `i` covers `j` in the designed
game. It is not a calibrated probability or structural-causal effect. The
implemented matrix combines:

- diagonal self-coverage;
- call-graph coverage rescue;
- time-series-derived trace propagation; and
- victim demotion applied to the self-coverage of a likely downstream victim.

For coalition `S subseteq \mathcal P`, the noisy-OR-inspired coverage transform is

```text
q_j(S) = 1 - product_{l in S} (1 - a_lj).
```

The characteristic function is weighted explained anomaly:

```text
v(S) = sum_{j=1}^N w_j q_j(S).
```

This is monotone for nonnegative `w_j` and bounded above by `sum_j w_j`.
It also has diminishing marginal coverage: once a target is well explained by
other players, another player's marginal contribution to that target shrinks.

MERIT uses the unnormalized Banzhaf value, i.e., the uniformly averaged
marginal contribution over coalitions not containing player `i`:

```text
beta_i = 2^{-(K-1)} sum_{S subseteq \mathcal P \\ {i}}
         [v(S union {i}) - v(S)].
```

"Unnormalized" here means that the values are not rescaled to sum to
`v(\mathcal P)`.
The final ranking code rescales positive service scores by their maximum only
for ranking; that is not a normalized Banzhaf index.

## Closed-form lemma

**Lemma.** For the weighted noisy-OR game above,

```text
beta_i = sum_{j=1}^N w_j a_ij
         product_{l in \mathcal P \\ {i}} (1 - a_lj / 2).
```

**Proof.** For a fixed target `j`, adding `i` to coalition `S` increases its
coverage by

```text
q_j(S union {i}) - q_j(S)
  = a_ij product_{l in S} (1 - a_lj).
```

Uniformly averaging over all subsets of `\mathcal P \\ {i}` is equivalent to including
each other player independently with probability one half. Independence gives

```text
E_S[product_{l in S} (1 - a_lj)]
  = product_{l != i} [(1/2) * 1 + (1/2) * (1 - a_lj)]
  = product_{l != i} (1 - a_lj / 2).
```

Multiplying by `w_j a_ij`, summing over targets, and using linearity of
expectation proves the expression. The brute-force validator enumerates the
same `2^(K-1)` coalitions per player and must agree within the frozen absolute
tolerance `1e-9` for every tested `K <= 15`. QED.

## Complexity

The closed form constructs `1 - a/2`, one product per target, and one weighted
sum per player. Its time complexity is `O(KN)`. The vectorized implementation
stores the coverage-strength matrix and a `K x N` product-without-player
temporary,
so its implemented space complexity is `O(KN)`; an algebraically equivalent
streaming implementation could use `O(N)` auxiliary space.

Direct Banzhaf enumeration evaluates `K * 2^(K-1)` player-coalition marginal
contributions. With cached or incrementally maintained coalition coverage,
this is `O(K 2^K N)` time and `O(N)` auxiliary space in addition to the input.
The deliberately simple validation implementation recomputes noisy-OR products
for each coalition and can incur an additional factor up to `K`; empirical
runtime claims refer to that measured implementation and never replace the
algorithmic bound with a projection.

The individual-attribution comparator uses the same players, targets, weights,
coverage-strength matrix, coverage, and victim factors, but assigns singleton
weighted coverage `sum_j w_j a_ij`. It is a controlled attribution replacement,
not a claim that Banzhaf is the only possible game-theoretic score.
