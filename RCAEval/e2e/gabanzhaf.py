import os
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from time import perf_counter

warnings.filterwarnings("ignore")

from RCAEval.io.time_series import preprocess


@dataclass(frozen=True)
class _GabanzhafConfig:
    """Centralized runtime configuration for the GA-Banzhaf pipeline."""

    top_k_svcs: int = 15
    dk_select_useful: bool = False
    exact_candidate_k: int = 12
    banzhaf_mode: str = "closed_form"
    causal_probability_lambda: float = 1.5
    early_window_ratio: float = 0.25
    early_window_min_len: int = 5

    @staticmethod
    def _positive_int_or_default(value, default):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return int(default)
        if not np.isfinite(parsed) or parsed <= 0.0:
            return int(default)
        parsed_int = int(parsed)
        if parsed_int <= 0:
            return int(default)
        return parsed_int

    @classmethod
    def from_kwargs(cls, kwargs):
        return cls(
            top_k_svcs=int(kwargs.get("top_k_svcs", cls.top_k_svcs)),
            dk_select_useful=bool(kwargs.get("dk_select_useful", cls.dk_select_useful)),
            exact_candidate_k=cls._positive_int_or_default(
                kwargs.get("exact_candidate_k", cls.exact_candidate_k),
                cls.exact_candidate_k,
            ),
            banzhaf_mode=str(kwargs.get("banzhaf_mode", cls.banzhaf_mode)),
            causal_probability_lambda=float(
                kwargs.get("causal_probability_lambda", cls.causal_probability_lambda)
            ),
            early_window_ratio=float(kwargs.get("early_window_ratio", cls.early_window_ratio)),
            early_window_min_len=cls._positive_int_or_default(
                kwargs.get("early_window_min_len", cls.early_window_min_len),
                cls.early_window_min_len,
            ),
        )


# ==========================================
# Internal helpers
# ==========================================


def _parse_svc(col):
    """Parse service name from a metric column (last underscore separator)."""
    parts = col.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else col


def _norm01_arr(a):
    a = np.asarray(a, dtype=float)
    lo = np.nanmin(a) if a.size else 0.
    hi = np.nanmax(a) if a.size else 1.
    return (a - lo) / (hi - lo + 1e-9)


def _norm_dict(d):
    lo, hi = min(d.values(), default=0.), max(d.values(), default=1.)
    span = hi - lo if hi > lo else 1.
    return {k: (v - lo) / span for k, v in d.items()}


def _metric_evidence(normal_mat, anomal_mat, cols, config=None):
    """Metric-level evidence from robust magnitude, persistence and peak.

    The score remains unsupervised: it only compares the post-injection window
    with the pre-injection baseline.  Percentile and persistence terms reduce
    the chance that a single noisy spike dominates the ranking.
    """
    if config is None:
        config = _GabanzhafConfig.from_kwargs({})
    normal_mat = np.asarray(normal_mat, dtype=float)
    anomal_mat = np.asarray(anomal_mat, dtype=float)
    med = np.nanmedian(normal_mat, axis=0)
    q75 = np.nanpercentile(normal_mat, 75, axis=0)
    q25 = np.nanpercentile(normal_mat, 25, axis=0)
    std = np.nanstd(normal_mat, axis=0)
    iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)

    rz = np.abs((anomal_mat - med) / iqr)
    rz = np.nan_to_num(rz, nan=0., posinf=0., neginf=0.)
    # Selective Winsorization: cap metrics where IQR=0 AND std≈0 (truly constant
    # baseline). These produce inflated rz through the unit denominator even for
    # tiny shifts, masking real causes. The std<=1e-6 guard preserves columns
    # that have IQR=0 but non-trivial spread so their rz is still meaningful.
    near_zero_baseline = ((q75 - q25) <= 0) & (std <= 1e-6)   # shape (n_cols,)
    rz = np.where(near_zero_baseline, np.clip(rz, 0., 10.), rz)

    anomal_len = rz.shape[0]
    early_len = max(int(round(anomal_len * config.early_window_ratio)), 1)
    use_full_for_early = early_len < config.early_window_min_len
    early_rz = rz if use_full_for_early else rz[:early_len]

    mag = _norm01_arr(np.nanpercentile(rz, 95, axis=0))
    peak = _norm01_arr(np.nanmax(rz, axis=0))
    persistence = _norm01_arr(np.mean(rz > 3.0, axis=0))
    full = _norm01_arr(0.55 * mag + 0.30 * persistence + 0.15 * peak)
    early_mag = _norm01_arr(np.nanpercentile(early_rz, 95, axis=0))
    early_peak = _norm01_arr(np.nanmax(early_rz, axis=0))
    early = full.copy() if use_full_for_early else _norm01_arr(0.70 * early_mag + 0.30 * early_peak)
    score = _norm01_arr(0.60 * early + 0.40 * full)

    return {col: float(score[i]) for i, col in enumerate(cols)}


def _service_evidence(cols, metric_ev):
    """Aggregate metric evidence into service evidence.

    Two complementary signals are blended:

    1. Aggregate (70 %): percentile-weighted blend of a service's own metric
       scores — the primary signal for clearly localised faults.

    2. Metric-type dominance (30 %): for each metric type (the suffix after
       the last underscore, e.g. "cpu", "mem", "latency") the service with the
       highest score earns a bonus proportional to its margin over the
       runner-up.  This is decisive for broadcast faults (e.g. CPU injection)
       where every service shows elevated metrics: the injected service still
       leads in the fault-relevant metric type even when absolute levels are
       similar across the system.

    No ground truth is used; no branching on system size.
    """
    svc_to_cols: dict = {}
    for col in cols:
        svc_to_cols.setdefault(_parse_svc(col), []).append(col)

    # ── 1. Per-service aggregate score ───────────────────────────────────────
    svc_agg: dict = {}
    for svc, svc_cols in svc_to_cols.items():
        arr = np.asarray([metric_ev[c] for c in svc_cols], dtype=float)
        top = np.sort(arr)[-min(3, len(arr)):]
        svc_agg[svc] = float(
            0.50 * np.percentile(arr, 95)
            + 0.30 * np.mean(top)
            + 0.20 * np.max(arr)
        )

    # ── 2. Metric-type dominance ─────────────────────────────────────────────
    # Track the top-2 scores per metric type across services.
    # "metric type" = everything after the service prefix  (e.g. "cpu", "mem").
    type_best: dict   = {}   # mtype -> (winning_svc, score)
    type_second: dict = {}   # mtype -> best score from a *different* service

    for col in cols:
        svc   = _parse_svc(col)
        mtype = col[len(svc) + 1:] if col.startswith(svc + "_") else col
        sc    = metric_ev[col]

        if mtype not in type_best or sc > type_best[mtype][1]:
            prev = type_best.get(mtype)
            if prev is not None and prev[0] != svc:
                # Previous winner (different service) becomes second-best candidate
                if prev[1] > type_second.get(mtype, 0.):
                    type_second[mtype] = prev[1]
            type_best[mtype] = (svc, sc)
        elif svc != type_best[mtype][0] and sc > type_second.get(mtype, 0.):
            type_second[mtype] = sc

    dom: dict = {svc: 0. for svc in svc_agg}
    for mtype, (best_svc, best_sc) in type_best.items():
        margin = best_sc - type_second.get(mtype, 0.)
        dom[best_svc] = dom.get(best_svc, 0.) + margin

    dom_max  = max(dom.values(), default=1.)
    dom_norm = {s: v / (dom_max + 1e-9) for s, v in dom.items()}

    # ── 3. Blend ─────────────────────────────────────────────────────────────
    svc_scores = {
        svc: 0.70 * svc_agg[svc] + 0.30 * dom_norm.get(svc, 0.)
        for svc in svc_agg
    }
    return _norm_dict(svc_scores)


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

    half_complements = 1.0 - 0.5 * prob                          # (K, N)
    product_all = np.prod(half_complements, axis=0)               # (N,)
    product_without = product_all / np.maximum(half_complements, 1e-12)  # (K, N)
    scores = np.sum(weights * prob * product_without, axis=1)     # (K,)
    return np.maximum(scores, 0.0)


def _bruteforce_probabilistic_banzhaf(prob_matrix, anomaly_weights):
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

    n_players = prob.shape[0]
    scores = np.zeros(n_players, dtype=float)
    players = list(range(n_players))
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        count = 0
        for mask in range(1 << len(others)):
            coalition = [
                player
                for bit, player in enumerate(others)
                if mask & (1 << bit)
            ]
            with_i = coalition + [i]
            total += float(
                np.dot(weights, _noisy_or_coverage(prob, with_i))
                - np.dot(weights, _noisy_or_coverage(prob, coalition))
            )
            count += 1
        scores[i] = total / max(count, 1)
    return np.maximum(scores, 0.0)


def _banzhaf_power(prob_matrix, anomaly_weights, config):
    if config.banzhaf_mode == "closed_form":
        return _probabilistic_banzhaf(prob_matrix, anomaly_weights)
    if config.banzhaf_mode == "bruteforce":
        return _bruteforce_probabilistic_banzhaf(prob_matrix, anomaly_weights)
    raise ValueError(f"Unsupported banzhaf_mode: {config.banzhaf_mode}")


def _build_topology(svc_names, svc_series, corr_thr=0.08, max_in=3):
    """Build weighted directed topology from asymmetric lag-1 correlations.

    Vectorised: stacks all available service series into a matrix and
    computes the full n_svc × n_svc lag-1 correlation matrix in one
    numpy matmul.  Edge confidence is asymmetric: u→v is strong only when
    u leads v more than v leads u.
    """
    svcs = [s for s in svc_names if svc_series.get(s) is not None]
    if len(svcs) < 2:
        return []

    T = min(len(svc_series[s]) for s in svcs) - 1
    if T < 3:
        return []

    # Stack: (n_svcs, T)
    X = np.array([svc_series[s][:T]   for s in svcs], dtype=float)  # current
    Y = np.array([svc_series[s][1:T+1] for s in svcs], dtype=float)  # lag-1

    # Z-score each row
    X -= X.mean(axis=1, keepdims=True)
    Y -= Y.mean(axis=1, keepdims=True)
    xn = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    yn = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9
    X /= xn;  Y /= yn

    # Lag-1 correlation matrix: C[i,j] = corr(X[i], Y[j]) = X_i leads Y_j.
    C = X @ Y.T  # (n_svcs, n_svcs)
    np.fill_diagonal(C, 0.)

    confidence = np.maximum(0., C - C.T)
    np.fill_diagonal(confidence, 0.)

    edges = []
    for j, v in enumerate(svcs):
        parents = np.where(confidence[:, j] > corr_thr)[0]
        if len(parents) > max_in:
            parents = parents[np.argsort(confidence[parents, j])[-max_in:]]
        for i in parents:
            edges.append((svcs[i], v, float(confidence[i, j])))
    return edges


def _split_raw_windows(data, inject_time, anomalies):
    if anomalies is not None:
        split_idx = int(anomalies[0])
        return data.head(split_idx), data.tail(len(data) - split_idx)
    if inject_time is not None and "time" in data.columns:
        return data[data["time"] < inject_time], data[data["time"] >= inject_time]
    split_idx = len(data) // 2
    return data.head(split_idx), data.tail(len(data) - split_idx)


def _preprocess_windows(normal_raw, anomal_raw, dataset, config):
    n_rows = len(normal_raw)
    full_proc = preprocess(
        pd.concat([normal_raw, anomal_raw], ignore_index=True),
        dataset,
        config.dk_select_useful,
    )

    normal_df = full_proc.iloc[:n_rows].reset_index(drop=True)
    anomal_df = full_proc.iloc[n_rows:].reset_index(drop=True)

    shared = [c for c in normal_df.columns if c in anomal_df.columns]
    normal_df = normal_df[shared]
    anomal_df = anomal_df[shared]

    min_len = min(len(normal_df), len(anomal_df))
    normal_df = normal_df.tail(min_len).reset_index(drop=True)
    anomal_df = anomal_df.head(min_len).reset_index(drop=True)
    no_time = pd.concat([normal_df, anomal_df], ignore_index=True).drop(
        columns=["time"], errors="ignore"
    )
    return normal_df, anomal_df, no_time


def _service_series_for_pool(cols, no_time, pool_svcs):
    values = no_time.values
    svc_series = {}
    for svc in pool_svcs:
        idxs = [i for i, col in enumerate(cols) if _parse_svc(col) == svc]
        if idxs:
            svc_series[svc] = values[:, idxs].max(axis=1).astype(float)
    return svc_series


def _initial_service_pool(svc_ranked, seed_svcs, config):
    # Pool = top services by anomaly score, widened to 2× top_k (or +5, whichever
    # is larger) so lag-correlation has enough context to find upstream edges.
    pool_k = min(len(svc_ranked), max(config.top_k_svcs * 2, config.top_k_svcs + 5))
    return set(svc_ranked[:pool_k]) | set(seed_svcs)


def _candidate_services(seed_svcs, lag_corr_edges, svc_names):
    """Expand seeds with their lag-correlation upstreams (anomaly explainers)."""
    upstream = {s: set() for s in svc_names}
    for u, v, _ in lag_corr_edges:
        upstream.get(v, set()).add(u)
    candidates = set(seed_svcs)
    for svc in list(candidates):
        candidates |= upstream.get(svc, set())
    return sorted(candidates)


def _select_exact_candidates(candidates, svc_anom_dict, config):
    k = max(1, int(config.exact_candidate_k))
    candidates = list(dict.fromkeys(candidates))
    return sorted(
        candidates,
        key=lambda svc: (-_finite_nonnegative(svc_anom_dict.get(svc, 0.0)), svc),
    )[:k]


def _causal_explanation_matrix(
    cand_list,
    service_list,
    svc_anom_dict,
    topology_edges,
    config,
):
    if len(cand_list) != len(set(cand_list)):
        raise ValueError("Duplicate candidate name")
    if len(service_list) != len(set(service_list)):
        raise ValueError("Duplicate service name")
    cand_idx = {svc: i for i, svc in enumerate(cand_list)}
    svc_idx = {svc: j for j, svc in enumerate(service_list)}
    matrix = np.zeros((len(cand_list), len(service_list)), dtype=float)
    lam = _finite_nonnegative(config.causal_probability_lambda)
    if lam <= 0.0:
        lam = 1.0

    for svc, i in cand_idx.items():
        if svc in svc_idx:
            matrix[i, svc_idx[svc]] = 1.0

    for u, v, w in topology_edges:
        if u not in cand_idx or v not in svc_idx:
            continue
        alignment = np.sqrt(
            _finite_nonnegative(svc_anom_dict.get(u, 0.0))
            * _finite_nonnegative(svc_anom_dict.get(v, 0.0))
        )
        raw = _finite_nonnegative(w) * alignment
        matrix[cand_idx[u], svc_idx[v]] = max(
            matrix[cand_idx[u], svc_idx[v]],
            1.0 - np.exp(-lam * raw),
        )

    return np.clip(matrix, 0.0, 1.0)


def _finite_nonnegative(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value) or value < 0.0:
        return 0.0
    return value


def _rank_metrics(all_cols, svc_banzhaf):
    """Rank metric columns by their service's Banzhaf score (column name breaks ties)."""
    finite_banzhaf = {
        svc: _finite_nonnegative(score)
        for svc, score in (svc_banzhaf or {}).items()
    }
    if not finite_banzhaf or len(set(finite_banzhaf.values())) == 1:
        nb = {svc: 1.0 for svc in {_parse_svc(col) for col in all_cols}}
    else:
        max_banzhaf = max(finite_banzhaf.values())
        nb = {svc: score / max_banzhaf for svc, score in finite_banzhaf.items()}

    return sorted(all_cols, key=lambda col: (-nb.get(_parse_svc(col), 0.0), col))


# ==========================================
# Core pipeline
# ==========================================


def _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs):
    """
    Pipeline: window split → metric evidence → service evidence
              → lag-1 correlation topology → Banzhaf → metric ranking
    """
    config = _GabanzhafConfig.from_kwargs(kwargs)
    return_stage_timing = bool(kwargs.get("return_stage_timing", False))
    stage_timing = {}
    last_tick = perf_counter()

    def _mark_stage(name):
        nonlocal last_tick
        if not return_stage_timing:
            return
        now = perf_counter()
        stage_timing[name] = float(now - last_tick)
        last_tick = now

    # ── 1. Window split ──────────────────────────────────────────────────────
    normal_raw, anomal_raw = _split_raw_windows(data, inject_time, anomalies)

    # ── 2. Preprocess jointly ────────────────────────────────────────────────
    normal_df, anomal_df, no_time = _preprocess_windows(
        normal_raw, anomal_raw, dataset, config
    )
    cols = list(no_time.columns)
    normal_mat = normal_df.drop(columns=["time"], errors="ignore").values.astype(float)
    anomal_mat = anomal_df.drop(columns=["time"], errors="ignore").values.astype(float)
    _mark_stage("preprocess")

    # ── 3. Metric and service evidence ───────────────────────────────────────
    metric_ev = _metric_evidence(normal_mat, anomal_mat, cols, config=config)
    svc_anom_dict = _service_evidence(cols, metric_ev)
    svc_names = sorted(svc_anom_dict.keys())
    _mark_stage("metric_evidence")

    # ── 4. Candidate-first bounded lag-correlation topology ──────────────────
    svc_ranked = sorted(svc_names, key=lambda s: svc_anom_dict.get(s, 0.), reverse=True)
    seed_svcs = set(svc_ranked[: config.top_k_svcs])

    pool_svcs = _initial_service_pool(svc_ranked, seed_svcs, config)
    svc_series_full = _service_series_for_pool(cols, no_time, pool_svcs)
    topology_edges = _build_topology(sorted(pool_svcs), svc_series_full)
    cand_list = _candidate_services(seed_svcs, topology_edges, svc_names)
    _mark_stage("topology")

    # ── 5. Probabilistic causal game ─────────────────────────────────────────
    game_candidates = _select_exact_candidates(cand_list, svc_anom_dict, config)
    game_services = sorted(svc_anom_dict.keys())
    prob_matrix = _causal_explanation_matrix(
        game_candidates,
        game_services,
        svc_anom_dict,
        topology_edges,
        config,
    )
    anomaly_weights = np.asarray(
        [svc_anom_dict.get(svc, 0.0) for svc in game_services],
        dtype=float,
    )

    # ── 6. Banzhaf power index ────────────────────────────────────────────────
    banzhaf_values = _banzhaf_power(prob_matrix, anomaly_weights, config)
    svc_banzhaf = {
        game_candidates[i]: float(banzhaf_values[i])
        for i in range(len(game_candidates))
    }
    _mark_stage("banzhaf")

    # ── 7. Final metric-level ranking by Banzhaf score ───────────────────────
    ranks = _rank_metrics(cols, svc_banzhaf)
    _mark_stage("ranking")

    result = {"ranks": ranks, "node_names": cols}
    if bool(kwargs.get("return_diagnostics", False)):
        result["game_candidates"] = game_candidates
        result["metric_service_evidence"] = dict(svc_anom_dict)
    if return_stage_timing:
        result["stage_timing"] = stage_timing
    return result


# ==========================================
# RCAEval entry point
# ==========================================

def gabanzhaf(data, inject_time=None, dataset=None, num_loop=None,
              sli=None, anomalies=None, **kwargs):
    """Pure probabilistic causal Banzhaf RCA.

    Pipeline: window split → metric evidence → service evidence →
    lag-1 correlation topology → probabilistic noisy-OR Banzhaf → rank by
    Banzhaf score.  Trace onset, call-graph and residual stages were removed
    after ablation showed them net-negative on re2 (see benchmark_*.json):
    the stripped pipeline matched or beat the full one while running ~30 %
    faster.

    Env knobs:
      GABANZHAF_BANZHAF_MODE   closed_form (default) | bruteforce
      GABANZHAF_EXACT_CANDIDATE_K   override the game candidate count
      GABANZHAF_PROFILE=1      print per-stage timing to stderr
                               (preprocess | metric_evidence | topology |
                                banzhaf | ranking)
    """
    import sys
    _profile = os.environ.get("GABANZHAF_PROFILE") == "1"

    runtime_kwargs = dict(kwargs)
    if os.environ.get("GABANZHAF_BANZHAF_MODE"):
        runtime_kwargs.setdefault("banzhaf_mode", os.environ["GABANZHAF_BANZHAF_MODE"])
    if os.environ.get("GABANZHAF_EXACT_CANDIDATE_K"):
        runtime_kwargs.setdefault(
            "exact_candidate_k",
            os.environ["GABANZHAF_EXACT_CANDIDATE_K"],
        )
    if _profile:
        runtime_kwargs["return_stage_timing"] = True

    result = _gabanzhaf_core(data, inject_time, dataset, anomalies, runtime_kwargs)

    if _profile and "stage_timing" in result:
        st = result["stage_timing"]
        total = sum(st.values())
        label = str(dataset) or "?"
        parts = [f"{k}={v:.3f}s" for k, v in st.items()] + [f"total={total:.3f}s"]
        print(f"[PROFILE] {label}  " + "  ".join(parts), file=sys.stderr)

    return result
