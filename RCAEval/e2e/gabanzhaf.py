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
    rank_mode: str = "banzhaf"
    early_window_ratio: float = 0.25
    early_window_min_len: int = 5
    # Call-graph coverage: rescue a quiet upstream root that explains several
    # loud downstream victims (the delay/loss propagation signature). Gated on
    # >=min_loud children above loud_thr so single-strong-anomaly faults
    # (cpu/mem/disk/socket) are left untouched.
    coverage_conf: float = 0.7
    coverage_loud_thr: float = 0.4
    coverage_lat_floor: float = 0.1
    coverage_min_loud: int = 2

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
            rank_mode=str(kwargs.get("rank_mode", cls.rank_mode)),
            early_window_ratio=float(kwargs.get("early_window_ratio", cls.early_window_ratio)),
            early_window_min_len=cls._positive_int_or_default(
                kwargs.get("early_window_min_len", cls.early_window_min_len),
                cls.early_window_min_len,
            ),
            coverage_conf=float(kwargs.get("coverage_conf", cls.coverage_conf)),
            coverage_loud_thr=float(kwargs.get("coverage_loud_thr", cls.coverage_loud_thr)),
            coverage_lat_floor=float(kwargs.get("coverage_lat_floor", cls.coverage_lat_floor)),
            coverage_min_loud=cls._positive_int_or_default(
                kwargs.get("coverage_min_loud", cls.coverage_min_loud),
                cls.coverage_min_loud,
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


def _load_call_graph(dataset):
    """Load the precomputed static service call graph for the dataset.

    Returns {caller: [callees]}. The graph is system-static, so it is built
    once offline (see tt_callgraph.json) and only loaded — no per-case trace
    parsing. Returns {} when no graph ships for the dataset.
    """
    import json
    name = str(dataset or "").lower()
    if "tt" not in name and "train-ticket" not in name:
        return {}
    path = os.path.join(os.path.dirname(__file__), "tt_callgraph.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _service_latency_scores(cols, metric_ev):
    """Per-service latency-metric score (max over its *_latency columns).

    Used to tell propagation victims (loud because latency spiked) from
    services with their own resource fault (loud because of cpu/mem/disk).
    """
    out = {}
    for col in cols:
        if col.rsplit("_", 1)[-1] != "latency":
            continue
        svc = _parse_svc(col)
        out[svc] = max(out.get(svc, 0.0), _finite_nonnegative(metric_ev.get(col, 0.0)))
    return out


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
    max_banzhaf = max(finite_banzhaf.values(), default=0.0)
    if max_banzhaf > 0.0:
        nb = {svc: score / max_banzhaf for svc, score in finite_banzhaf.items()}
    else:
        nb = {svc: 1.0 for svc in {_parse_svc(col) for col in all_cols}}

    return sorted(all_cols, key=lambda col: (-nb.get(_parse_svc(col), 0.0), col))


def _rank_evidence_metrics(all_cols, metric_ev, svc_anom_dict, rank_mode):
    """Rank metrics without the causal game or Banzhaf power calculation."""
    if rank_mode == "service_evidence":
        return sorted(
            all_cols,
            key=lambda col: (
                -_finite_nonnegative(svc_anom_dict.get(_parse_svc(col), 0.0)),
                -_finite_nonnegative(metric_ev.get(col, 0.0)),
                col,
            ),
        )
    if rank_mode == "metric_evidence":
        return sorted(
            all_cols,
            key=lambda col: (
                -_finite_nonnegative(metric_ev.get(col, 0.0)),
                col,
            ),
        )
    raise ValueError(f"Unsupported evidence rank mode: {rank_mode}")


# ==========================================
# Core pipeline
# ==========================================


def _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs):
    """
    Pipeline: window split → metric evidence → service evidence
              → call-graph coverage → Banzhaf → metric ranking
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

    if config.rank_mode not in {"banzhaf", "service_evidence", "metric_evidence"}:
        raise ValueError(f"Unsupported rank_mode: {config.rank_mode}")

    # ── 4. Candidates = top services by anomaly score ────────────────────────
    svc_ranked = sorted(svc_names, key=lambda s: svc_anom_dict.get(s, 0.), reverse=True)
    seed_svcs = set(svc_ranked[: config.top_k_svcs])
    cand_list = sorted(seed_svcs)
    _mark_stage("candidates")

    # ── 4b. Call-graph coverage: rescue quiet roots of multi-victim faults ────
    if bool(kwargs.get("use_coverage", True)):
        call_graph = _load_call_graph(dataset)
        svc_latency = _service_latency_scores(cols, metric_ev)
        coverage_parents = _coverage_parents(call_graph, svc_anom_dict, svc_latency, config)
    else:
        coverage_parents = {}

    # ── 5. Probabilistic causal game ─────────────────────────────────────────
    game_candidates = _select_exact_candidates(cand_list, svc_anom_dict, config)
    # Force qualified coverage parents into the game even when their own
    # anomaly is too low to make the anomaly-ranked top-k cut.
    for parent in coverage_parents:
        if parent in svc_anom_dict and parent not in game_candidates:
            game_candidates.append(parent)
    game_services = sorted(svc_anom_dict.keys())
    if config.rank_mode == "banzhaf":
        prob_matrix = _causal_explanation_matrix(game_candidates, game_services)
        prob_matrix = _apply_coverage_edges(
            prob_matrix,
            game_candidates,
            game_services,
            coverage_parents,
            config.coverage_conf,
        )
        anomaly_weights = np.asarray(
            [svc_anom_dict.get(svc, 0.0) for svc in game_services],
            dtype=float,
        )
    else:
        prob_matrix = np.zeros((0, 0), dtype=float)
        anomaly_weights = np.zeros(0, dtype=float)

    # ── 6. Banzhaf power index ────────────────────────────────────────────────
    svc_banzhaf = {}
    if config.rank_mode == "banzhaf":
        banzhaf_values = _banzhaf_power(prob_matrix, anomaly_weights, config)
        svc_banzhaf = {
            game_candidates[i]: float(banzhaf_values[i])
            for i in range(len(game_candidates))
        }
    _mark_stage("banzhaf")

    # ── 7. Final metric-level ranking by Banzhaf score ───────────────────────
    if config.rank_mode == "banzhaf":
        ranks = _rank_metrics(cols, svc_banzhaf)
    else:
        ranks = _rank_evidence_metrics(
            cols,
            metric_ev,
            svc_anom_dict,
            config.rank_mode,
        )
    _mark_stage("ranking")

    result = {"ranks": ranks, "node_names": cols}
    if bool(kwargs.get("return_diagnostics", False)):
        diagnostics = {
            "rank_mode": config.rank_mode,
            "candidate_pool": list(cand_list),
            "game_candidates": list(game_candidates),
            "game_services": list(game_services),
            "causal_probability_matrix": prob_matrix.tolist(),
            "metric_evidence": dict(metric_ev),
            "metric_service_evidence": dict(svc_anom_dict),
            "service_banzhaf": dict(svc_banzhaf),
            "root_service": kwargs.get("root_service"),
            "root_candidate_recall": (
                kwargs.get("root_service") in set(game_candidates)
                if kwargs.get("root_service") is not None
                else None
            ),
        }
        result["diagnostics"] = diagnostics
        result.update(diagnostics)
    if return_stage_timing:
        result["stage_timing"] = stage_timing
    return result


# ==========================================
# RCAEval entry point
# ==========================================

def gabanzhaf(data, inject_time=None, dataset=None, num_loop=None,
              sli=None, anomalies=None, **kwargs):
    """Probabilistic causal Banzhaf RCA with call-graph coverage.

    Pipeline: window split → metric evidence → service evidence →
    static call-graph coverage (rescues quiet upstream roots of delay/loss
    propagation) → probabilistic noisy-OR Banzhaf → rank by Banzhaf score.
    Lag-1 correlation topology was removed after ablation showed it dead on
    re2-{tt,ob,ss} and re1-{ss,tt} and net-negative on re1-ob.

    Env knobs:
      GABANZHAF_BANZHAF_MODE   closed_form (default) | bruteforce
      GABANZHAF_EXACT_CANDIDATE_K   override the game candidate count
      GABANZHAF_DISABLE_COVERAGE=1  turn off call-graph coverage (ablation)
      GABANZHAF_PROFILE=1      print per-stage timing to stderr
                               (preprocess | metric_evidence | candidates |
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
    if os.environ.get("GABANZHAF_DISABLE_COVERAGE") == "1":
        runtime_kwargs.setdefault("use_coverage", False)
    if os.environ.get("GABANZHAF_RANK_MODE"):
        runtime_kwargs.setdefault("rank_mode", os.environ["GABANZHAF_RANK_MODE"])
    if os.environ.get("GABANZHAF_WRITE_DIAGNOSTICS") == "1":
        runtime_kwargs["return_diagnostics"] = True
        runtime_kwargs["return_stage_timing"] = True
    args_ns = kwargs.get("args")
    if args_ns and hasattr(args_ns, "data_path"):
        case_dir = os.path.dirname(args_ns.data_path)
        fault_dir = os.path.basename(os.path.dirname(case_dir))
        runtime_kwargs["root_service"] = fault_dir.split("_", 1)[0]
    if _profile:
        runtime_kwargs["return_stage_timing"] = True

    result = _gabanzhaf_core(data, inject_time, dataset, anomalies, runtime_kwargs)
    if "diagnostics" in result:
        result["diagnostics"]["stage_timing"] = dict(
            result.get("stage_timing", {})
        )

    if _profile and "stage_timing" in result:
        st = result["stage_timing"]
        total = sum(st.values())
        label = str(dataset) or "?"
        parts = [f"{k}={v:.3f}s" for k, v in st.items()] + [f"total={total:.3f}s"]
        print(f"[PROFILE] {label}  " + "  ".join(parts), file=sys.stderr)

    return result
