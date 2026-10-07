"""Coverage-game construction and attribution primitives for MERIT."""

import numpy as np

from RCAEval.e2e.merit_evidence import _finite_nonnegative


def _noisy_or_coverage(prob_matrix, coalition):
    prob = np.nan_to_num(
        np.asarray(prob_matrix, dtype=float),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    prob = np.clip(prob, 0.0, 1.0)
    if prob.ndim != 2:
        raise ValueError("prob_matrix must be a 2-D array")

    try:
        coalition_arr = np.asarray(list(coalition))
    except TypeError as exc:
        raise ValueError("coalition must be a 1-D integer index array") from exc
    if coalition_arr.ndim != 1:
        raise ValueError("coalition must be a 1-D integer index array")
    if coalition_arr.size == 0:
        return np.zeros(prob.shape[1], dtype=float)

    if not np.issubdtype(coalition_arr.dtype, np.integer):
        raise ValueError("coalition must contain integer indices")

    coalition_idx = coalition_arr.astype(int)
    if np.any((coalition_idx < 0) | (coalition_idx >= prob.shape[0])):
        raise ValueError("coalition indices must be in [0, n_rows)")

    rows = prob[np.unique(coalition_idx)]
    return 1.0 - np.prod(1.0 - rows, axis=0)


def _probabilistic_banzhaf(prob_matrix, anomaly_weights):
    prob = np.nan_to_num(
        np.asarray(prob_matrix, dtype=float),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    prob = np.clip(prob, 0.0, 1.0)
    weights = np.nan_to_num(
        np.asarray(anomaly_weights, dtype=float).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    weights = np.maximum(weights, 0.0)
    if prob.ndim != 2:
        raise ValueError("prob_matrix must be a 2-D array")
    if prob.shape[1] != weights.shape[0]:
        raise ValueError("anomaly_weights length must match prob_matrix columns")
    if prob.shape[0] == 0:
        return np.zeros(0, dtype=float)

    # Delegate the half-complement/product-without-self computation to
    # _probabilistic_banzhaf_contributions (prob/weights are already cleaned
    # above, so its own nan_to_num/clip are no-ops here) and sum per-target
    # terms into per-candidate scores. Every term is already >= 0 (weights,
    # prob, and product_without are all non-negative), so this is numerically
    # identical to the previous inline computation.
    contributions = _probabilistic_banzhaf_contributions(prob, weights)
    return np.sum(contributions, axis=1)


def _probabilistic_banzhaf_contributions(prob_matrix, anomaly_weights):
    """Return the player-by-target terms whose row sums are Banzhaf scores."""

    prob = np.clip(
        np.nan_to_num(np.asarray(prob_matrix, dtype=float)), 0.0, 1.0
    )
    weights = np.maximum(
        np.nan_to_num(np.asarray(anomaly_weights, dtype=float).reshape(-1)), 0.0
    )
    if prob.ndim != 2 or prob.shape[1] != weights.shape[0]:
        raise ValueError("contribution dimensions do not match")
    if prob.shape[0] == 0:
        return np.zeros_like(prob)
    half_complements = 1.0 - 0.5 * prob
    product_all = np.prod(half_complements, axis=0)
    product_without = product_all / np.maximum(half_complements, 1e-12)
    return np.maximum(weights[None, :] * prob * product_without, 0.0)


def _edge_removal_faithfulness(prob_matrix, anomaly_weights):
    """Measure full-coalition utility loss after removing each explanation edge."""

    prob = np.clip(
        np.nan_to_num(np.asarray(prob_matrix, dtype=float)), 0.0, 1.0
    )
    weights = np.maximum(
        np.nan_to_num(np.asarray(anomaly_weights, dtype=float).reshape(-1)), 0.0
    )
    if prob.ndim != 2 or prob.shape[1] != weights.shape[0]:
        raise ValueError("faithfulness dimensions do not match")
    if prob.shape[0] == 0:
        return np.zeros_like(prob)
    complements = 1.0 - prob
    zero_mask = complements <= 1e-15
    zero_count = np.sum(zero_mask, axis=0)
    nonzero_product = np.prod(np.where(zero_mask, 1.0, complements), axis=0)
    product_without = np.zeros_like(prob)
    no_zero = zero_count == 0
    if np.any(no_zero):
        product_without[:, no_zero] = (
            nonzero_product[None, no_zero] / complements[:, no_zero]
        )
    one_zero = zero_count == 1
    if np.any(one_zero):
        product_without[:, one_zero] = np.where(
            zero_mask[:, one_zero],
            nonzero_product[None, one_zero],
            0.0,
        )
    return np.maximum(weights[None, :] * prob * product_without, 0.0)


def _coverage_parents(call_graph, svc_anom_dict, svc_latency, config):
    """Parents that explain >=min_loud propagation victims (delay/loss).

    A child counts as a victim when it is loud overall (anomaly >= loud_thr)
    AND shows at least some latency signal (>= lat_floor). The latency floor
    excludes a child that is loud purely from its own resource fault
    (cpu/mem/disk, latency ~0) — that child is itself a root, so its parent
    must not steal its #1. Returns {parent: [victim_child, ...]}.
    """
    loud_thr = _finite_nonnegative(config.coverage_loud_thr)
    lat_floor = _finite_nonnegative(config.coverage_lat_floor)
    min_loud = max(1, int(config.coverage_min_loud))
    parents = {}
    for caller, callees in (call_graph or {}).items():
        loud = [
            c for c in callees
            if svc_anom_dict.get(c, 0.0) >= loud_thr
            and svc_latency.get(c, 0.0) >= lat_floor
        ]
        if len(loud) >= min_loud:
            parents[caller] = loud
    return parents


def _apply_coverage_edges(matrix, cand_list, service_list, coverage_parents, conf):
    """Overlay parent->loud-child coverage with fixed confidence.

    Bypasses the alignment damping in _causal_explanation_matrix (which zeroes
    edges from a quiet parent) so a low-anomaly root still gets credited for
    covering its loud victims.
    """
    cand_idx = {svc: i for i, svc in enumerate(cand_list)}
    svc_idx = {svc: j for j, svc in enumerate(service_list)}
    conf = float(np.clip(conf, 0.0, 1.0))
    for parent, children in coverage_parents.items():
        i = cand_idx.get(parent)
        if i is None:
            continue
        for child in children:
            j = svc_idx.get(child)
            if j is not None:
                matrix[i, j] = max(matrix[i, j], conf)
    return matrix


def _apply_trace_topology_edges(matrix, cand_list, service_list, trace_edges):
    cand_idx = {svc: i for i, svc in enumerate(cand_list)}
    svc_idx = {svc: j for j, svc in enumerate(service_list)}
    for source, target, confidence, _source_kind, _lag in trace_edges:
        i = cand_idx.get(source)
        j = svc_idx.get(target)
        if i is None or j is None:
            continue
        matrix[i, j] = max(matrix[i, j], float(np.clip(confidence, 0.0, 1.0)))
    return matrix


def _causal_edges(coverage_parents, trace_edges, coverage_conf):
    """Merge coverage + trace edges (cause->effect); consumed by victim demotion."""
    edges = []
    conf = float(np.clip(coverage_conf, 0.0, 1.0))
    for parent, children in (coverage_parents or {}).items():
        for child in children:
            edges.append((parent, child, conf, "coverage", 0))
    for source, target, confidence, source_kind, lag in (trace_edges or []):
        edges.append((
            source,
            target,
            float(np.clip(confidence, 0.0, 1.0)),
            source_kind,
            int(lag),
        ))
    return edges


def _victim_factors(causal_edges, svc_anom_dict, config, protected=None):
    """Per-service self-coverage multiplier ∈ [victim_floor, 1].

    Edges are cause->effect. A service is a propagation victim when one of its
    causal predecessors is itself abnormal — then its own anomaly is credited
    upstream, so its self-coverage (diagonal 1.0) is shrunk toward victim_floor.
    A service whose predecessors are all normal/absent is a root and keeps 1.0.

    `protected` services are immune to demotion: coverage already inferred them
    to be roots that uniquely explain a cluster of loud victims, so the (opposite
    -direction) trace evidence must not overrule that and demote them as victims.
    """
    protected = set(protected or ())
    floor = float(np.clip(config.victim_floor, 0.0, 1.0))
    abnormal_thr = _finite_nonnegative(config.frontier_abnormal_threshold)
    normal_thr = min(abnormal_thr, _finite_nonnegative(config.frontier_normal_threshold))
    edge_thr = _finite_nonnegative(config.frontier_edge_threshold)
    uncertain_factor = float(np.clip(config.frontier_uncertain_stop_factor, 0.0, 1.0))

    predecessors = {}
    for source, target, confidence, _kind, _lag in causal_edges or []:
        if float(np.clip(confidence, 0.0, 1.0)) < edge_thr:
            continue
        predecessors.setdefault(str(target), []).append(str(source))

    factors = {}
    for svc, upstreams in predecessors.items():
        if svc in protected:
            continue  # coverage-inferred root → keep self-coverage at 1.0
        up_scores = [_finite_nonnegative(svc_anom_dict.get(u, 0.0)) for u in upstreams]
        if any(s >= abnormal_thr for s in up_scores):
            factors[svc] = floor
        elif any(s > normal_thr for s in up_scores):
            factors[svc] = floor + (1.0 - floor) * uncertain_factor
        # all-normal predecessors → root → leave at default 1.0 (omit)
    return factors


def _select_exact_candidates(candidates, svc_anom_dict, config):
    k = max(1, int(config.exact_candidate_k))
    candidates = list(dict.fromkeys(candidates))
    return sorted(
        candidates,
        key=lambda svc: (-_finite_nonnegative(svc_anom_dict.get(svc, 0.0)), svc),
    )[:k]


def _causal_explanation_matrix(cand_list, service_list):
    """Base explanation matrix: each candidate explains its own service (1.0).

    Off-diagonal coverage edges are overlaid separately by
    _apply_coverage_edges.
    """
    if len(cand_list) != len(set(cand_list)):
        raise ValueError("Duplicate candidate name")
    if len(service_list) != len(set(service_list)):
        raise ValueError("Duplicate service name")
    svc_idx = {svc: j for j, svc in enumerate(service_list)}
    matrix = np.zeros((len(cand_list), len(service_list)), dtype=float)
    for i, svc in enumerate(cand_list):
        if svc in svc_idx:
            matrix[i, svc_idx[svc]] = 1.0
    return matrix




def _filter_trace_topology_edges(edges, latency_family, minimum):
    """Keep trace edges whose target carries sufficient latency evidence."""
    return [edge for edge in edges if latency_family.get(edge[1], 0.0) >= minimum]


def _augment_coverage_candidates(candidates, coverage_parents, service_evidence):
    """Add qualified quiet coverage parents to the attribution game."""
    augmented = list(candidates)
    for parent in coverage_parents:
        if parent in service_evidence and parent not in augmented:
            augmented.append(parent)
    return augmented

