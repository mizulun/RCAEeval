"""Metric preprocessing, evidence extraction, and ranking for MERIT."""

import numpy as np
import pandas as pd

from RCAEval.e2e.merit_config import _GabanzhafConfig
from RCAEval.io.time_series import preprocess


def _cusum_metric_evidence(normal_mat, anomal_mat, cols, config):
    """Late-bound alternative scorer keeps the evidence module cycle-free."""
    from RCAEval.e2e.merit_ablation import _cusum_metric_evidence as scorer

    return scorer(normal_mat, anomal_mat, cols, config)


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

    config.scorer selects the underlying anomaly scorer: "robust_z" (default)
    keeps this function's original magnitude/persistence/peak behaviour
    exactly as-is; "cusum" routes to _cusum_metric_evidence instead.

    config.metric_zero_iqr_policy selects the robust-z treatment of columns
    whose normal-period IQR is zero: "unit" (default, historical denominator
    1), "std" (positive normal-period population std, else 1), or "exclude"
    (zero rz for those columns after the near-constant cap). Only
    normal-period statistics decide which columns are affected.
    """
    if config is None:
        config = _GabanzhafConfig.from_kwargs({})
    normal_mat = np.asarray(normal_mat, dtype=float)
    anomal_mat = np.asarray(anomal_mat, dtype=float)
    if getattr(config, "scorer", "robust_z") == "cusum":
        return _cusum_metric_evidence(normal_mat, anomal_mat, cols, config)
    zero_iqr_policy = getattr(config, "metric_zero_iqr_policy", "unit")
    med = np.nanmedian(normal_mat, axis=0)
    q75 = np.nanpercentile(normal_mat, 75, axis=0)
    q25 = np.nanpercentile(normal_mat, 25, axis=0)
    std = np.nanstd(normal_mat, axis=0)
    if zero_iqr_policy == "std":
        raw_iqr = q75 - q25
        iqr = np.where(raw_iqr > 0, raw_iqr, np.where(std > 0, std, 1.))
    else:
        # "unit" and "exclude" share the historical denominator; research
        # scripts replace this exact line via inspect.getsource.
        iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)

    rz = np.abs((anomal_mat - med) / iqr)
    rz = np.nan_to_num(rz, nan=0., posinf=0., neginf=0.)
    # Selective Winsorization: cap metrics where IQR=0 AND std≈0 (truly constant
    # baseline). These produce inflated rz through the unit denominator even for
    # tiny shifts, masking real causes. The std<=1e-6 guard preserves columns
    # that have IQR=0 but non-trivial spread so their rz is still meaningful.
    near_zero_baseline = ((q75 - q25) <= 0) & (std <= 1e-6)   # shape (n_cols,)
    rz = np.where(near_zero_baseline, np.clip(rz, 0., 10.), rz)
    if zero_iqr_policy == "exclude":
        # Zero-evidence imputation, not column removal: the columns stay in
        # every summary, normalization and service aggregation below.
        rz[:, (q75 - q25) <= 0] = 0.

    anomal_len = rz.shape[0]
    early_len = max(int(round(anomal_len * config.early_window_ratio)), 1)
    use_full_for_early = early_len < config.early_window_min_len
    early_rz = rz if use_full_for_early else rz[:early_len]

    mag = _norm01_arr(np.nanpercentile(rz, 95, axis=0))
    peak = _norm01_arr(np.nanmax(rz, axis=0))
    persistence = _norm01_arr(np.mean(rz > 3.0, axis=0))
    w_full = _norm_weights(
        (config.ev_w_mag, config.ev_w_persistence, config.ev_w_peak), (0.55, 0.30, 0.15)
    )
    full = _norm01_arr(w_full[0] * mag + w_full[1] * persistence + w_full[2] * peak)
    early_mag = _norm01_arr(np.nanpercentile(early_rz, 95, axis=0))
    early_peak = _norm01_arr(np.nanmax(early_rz, axis=0))
    w_early = _norm_weights((config.early_w_mag, config.early_w_peak), (0.70, 0.30))
    early = (
        full.copy()
        if use_full_for_early
        else _norm01_arr(w_early[0] * early_mag + w_early[1] * early_peak)
    )
    w_fuse = _norm_weights((config.fusion_w_early, config.fusion_w_full), (0.60, 0.40))
    score = (
        _norm01_arr(w_fuse[0] * early + w_fuse[1] * full)
        if config.use_early_full_fusion
        else full
    )

    return {col: float(score[i]) for i, col in enumerate(cols)}


def _norm_weights(values, fallback):
    """Renormalise a weight group to sum to 1 so any non-negative input is legal."""
    arr = np.asarray(
        [float(v) if np.isfinite(v) and v > 0 else 0.0 for v in values], dtype=float
    )
    total = arr.sum()
    return arr / total if total > 0 else np.asarray(fallback, dtype=float)


def _service_evidence(cols, metric_ev, use_metric_type_dominance=True, config=None):
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
    w_agg = (
        _norm_weights((config.agg_w_p95, config.agg_w_top3, config.agg_w_max),
                      (0.50, 0.30, 0.20))
        if config is not None
        else np.asarray((0.50, 0.30, 0.20))
    )
    svc_agg: dict = {}
    for svc, svc_cols in svc_to_cols.items():
        arr = np.asarray([metric_ev[c] for c in svc_cols], dtype=float)
        top = np.sort(arr)[-min(3, len(arr)):]
        svc_agg[svc] = float(
            w_agg[0] * np.percentile(arr, 95)
            + w_agg[1] * np.mean(top)
            + w_agg[2] * np.max(arr)
        )

    if not use_metric_type_dominance:
        return _norm_dict(svc_agg)

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
    w_svc = (
        _norm_weights((config.svc_w_agg, config.svc_w_dominance), (0.70, 0.30))
        if config is not None
        else np.asarray((0.70, 0.30))
    )
    svc_scores = {
        svc: w_svc[0] * svc_agg[svc] + w_svc[1] * dom_norm.get(svc, 0.)
        for svc in svc_agg
    }
    return _norm_dict(svc_scores)


def _service_metric_family_scores(cols, metric_ev, matches):
    """Shared implementation: per-service max score over columns matching ``matches``."""
    out = {}
    for col in cols:
        metric_name = col.rsplit("_", 1)[-1]
        if not matches(metric_name):
            continue
        svc = _parse_svc(col)
        out[svc] = max(out.get(svc, 0.0), _finite_nonnegative(metric_ev.get(col, 0.0)))
    return out


def _service_latency_scores(cols, metric_ev):
    """Per-service latency-metric score (max over its *_latency columns).

    Used to tell propagation victims (loud because latency spiked) from
    services with their own resource fault (loud because of cpu/mem/disk).
    """
    return _service_metric_family_scores(cols, metric_ev, lambda name: name == "latency")


def _service_latency_family_scores(cols, metric_ev):
    """Per-service score for latency and latency percentile metrics."""
    return _service_metric_family_scores(
        cols, metric_ev, lambda name: name.startswith("latency")
    )


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

    # The v14 headline estimand uses the complete case record. Keep every
    # available pre- and post-injection row; metric evidence supports unequal
    # baseline and observation lengths. Fixed equal-window truncation belongs
    # only to the explicitly separate window-sensitivity jobs.
    normal_df = normal_df.reset_index(drop=True)
    anomal_df = anomal_df.reset_index(drop=True)
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


def _rank_metrics(all_cols, svc_banzhaf, metric_evidence=None):
    """Rank by responsibility, then evidence, with stable schema-order ties."""
    finite_banzhaf = {
        svc: _finite_nonnegative(score)
        for svc, score in (svc_banzhaf or {}).items()
    }
    max_banzhaf = max(finite_banzhaf.values(), default=0.0)
    if max_banzhaf > 0.0:
        nb = {svc: score / max_banzhaf for svc, score in finite_banzhaf.items()}
    else:
        nb = {svc: 1.0 for svc in {_parse_svc(col) for col in all_cols}}

    if metric_evidence is None:
        return sorted(
            all_cols,
            key=lambda col: (-nb.get(_parse_svc(col), 0.0), col),
        )
    evidence = metric_evidence
    input_order = {column: index for index, column in enumerate(all_cols)}
    return sorted(
        all_cols,
        key=lambda col: (
            -nb.get(_parse_svc(col), 0.0),
            -_finite_nonnegative(evidence.get(col, 0.0)),
            input_order[col],
        ),
    )


def _service_ranks_from_metric_ranks(metric_ranks):
    """Collapse RCAEval's metric-level output schema to a service ranking."""
    service_ranks = []
    seen = set()
    for metric in metric_ranks:
        service = _parse_svc(metric)
        if service not in seen:
            seen.add(service)
            service_ranks.append(service)
    return service_ranks


