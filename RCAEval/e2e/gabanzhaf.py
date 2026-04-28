import warnings
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings("ignore")

from RCAEval.io.time_series import preprocess


# ==========================================
# Internal helpers
# ==========================================

def _parse_svc(col):
    """Parse service name from a metric column (last underscore separator)."""
    parts = col.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else col


def _redundancy_filtering(df, max_features=150, anom_scores=None):
    """Two-stage dimensionality reduction.

    Stage 1 — Pre-filter by anomaly score: if n_cols > 2×max_features,
    keep only the top-2×max_features most anomalous columns.  This
    guarantees the highest-signal metrics survive before clustering.

    Stage 2 — Agglomerative clustering to max_features: each cluster's
    representative is the column with the highest anomaly score.
    """
    df = df.dropna(axis=1)
    if df.shape[1] <= max_features:
        return df

    # Stage 1: anomaly-score pre-filter
    if anom_scores is not None and df.shape[1] > max_features * 2:
        top_k = max_features * 2
        ranked = sorted(df.columns, key=lambda c: anom_scores.get(c, 0.), reverse=True)
        df = df[ranked[:top_k]]

    # Stage 2: agglomerative clustering on transposed data (each column = sample)
    # ward linkage on raw features is O(n log n) and avoids computing the full
    # n×n distance matrix — much faster than precomputed complete-linkage.
    labels = AgglomerativeClustering(
        n_clusters=max_features, linkage="ward"
    ).fit_predict(df.values.T)
    if anom_scores is not None:
        selected = [
            max(df.columns[labels == k], key=lambda c: anom_scores.get(c, 0.))
            for k in np.unique(labels)
        ]
    else:
        selected = [df.columns[labels == k][0] for k in np.unique(labels)]
    return df[selected]


def _eval_fitness_vec(pop, anom_scores, denom, corr_matrix, num_metrics,
                      alpha_mmr, penalty_weight):
    """Vectorised fitness evaluation — no Python loop over individuals.

    coverage   = (pop @ anom_scores) / denom                      O(P×M)
    quad_form  = (pop @ corr_matrix * pop).sum(axis=1)            O(P×M²) numpy
    redundancy = (quad_form - sel_size) / max(sel_size*(sel_size-1), 1)
    sparsity   = sel_size * penalty_weight / num_metrics
    """
    sel_size = pop.sum(axis=1)                                    # (P,)
    coverage = (pop @ anom_scores) / denom                        # (P,)
    quad     = (pop @ corr_matrix * pop).sum(axis=1)              # (P,)
    denom_r  = np.maximum(sel_size * (sel_size - 1), 1.)
    redundancy = np.where(sel_size > 1, (quad - sel_size) / denom_r, 0.)
    sparsity = sel_size * penalty_weight / max(1, num_metrics)
    fitness = coverage - alpha_mmr * redundancy - sparsity
    fitness[sel_size == 0] = 0.
    return fitness


def _approximate_banzhaf_ga(normal_arr, anomal_arr, n_generations=8,
                            pop_size=20, ema_alpha=0.2, rng=None):
    """GA-Banzhaf — fully vectorised fitness + Banzhaf marginal estimation."""
    if rng is None:
        rng = np.random.default_rng(42)

    med = np.median(normal_arr, axis=0)
    iqr = np.percentile(normal_arr, 75, axis=0) - np.percentile(normal_arr, 25, axis=0)
    sc = np.where(iqr > 0, iqr, 1.)
    z_an = np.clip(np.abs((anomal_arr - med) / sc), 0., 100.)
    anom_scores = z_an.max(axis=0)
    denom = max(1., float(anom_scores.sum()))

    # Use numpy corrcoef — faster than pd.DataFrame.corr()
    if anomal_arr.shape[1] > 1:
        corr_matrix = np.abs(np.corrcoef(anomal_arr.T))
        np.nan_to_num(corr_matrix, nan=0., copy=False)
    else:
        corr_matrix = np.ones((1, 1))

    num_metrics = normal_arr.shape[1]
    ema_banzhaf = np.zeros(num_metrics)

    alpha_mmr   = 0.15
    elite_k     = max(2, pop_size // 10)
    p_mut_base  = 1.0 / max(1, num_metrics)
    warmup_gens = n_generations // 3

    pop = (rng.random((pop_size, num_metrics)) < 0.5).astype(np.float32)

    for gen in range(n_generations):
        penalty_weight = min(0.1, 0.05 * (1.0 + gen / n_generations))

        fitness = _eval_fitness_vec(pop, anom_scores, denom, corr_matrix,
                                    num_metrics, alpha_mmr, penalty_weight)

        # ── Banzhaf marginal — vectorised over all metrics simultaneously ──
        w_count  = pop.sum(axis=0)                               # (M,)
        wo_count = pop_size - w_count                            # (M,)
        w_sum    = (pop * fitness[:, None]).sum(axis=0)          # (M,)
        wo_sum   = ((1. - pop) * fitness[:, None]).sum(axis=0)  # (M,)
        w_mean   = np.where(w_count  > 0, w_sum  / np.maximum(w_count,  1.), 0.)
        wo_mean  = np.where(wo_count > 0, wo_sum / np.maximum(wo_count, 1.), 0.)
        current  = np.maximum(0., w_mean - wo_mean)
        ema_banzhaf = ema_alpha * current + (1 - ema_alpha) * ema_banzhaf

        # ── Elitism ──
        elite_idx = np.argsort(fitness)[-elite_k:]
        next_pop  = list(pop[elite_idx])

        # ── Guided Mutation parameters ──
        if gen >= warmup_gens and ema_banzhaf.max() > 0:
            b_norm = ((ema_banzhaf - ema_banzhaf.min()) /
                      (ema_banzhaf.max() - ema_banzhaf.min() + 1e-9))
            p_flip = p_mut_base * (1.0 + (1.0 - b_norm))
        else:
            p_flip = np.full(num_metrics, p_mut_base * 2)

        # ── Batch crossover + mutation (generate all offspring at once) ──
        n_offspring = pop_size - elite_k
        p1_idx = rng.integers(0, pop_size, size=n_offspring)
        p2_idx = rng.integers(0, pop_size, size=n_offspring)
        mask   = rng.random((n_offspring, num_metrics)) < 0.5
        children = np.where(mask, pop[p1_idx], pop[p2_idx]).copy()
        flip   = rng.random((n_offspring, num_metrics)) < p_flip
        children[flip] = 1.0 - children[flip]
        # Ensure at least one feature selected per child
        empty  = children.sum(axis=1) == 0
        if empty.any():
            fix_idx = rng.integers(num_metrics, size=empty.sum())
            children[np.where(empty)[0], fix_idx] = 1.0
        next_pop.extend(list(children))

        pop = np.array(next_pop[:pop_size], dtype=np.float32)

    return ema_banzhaf


def _build_topology(svc_names, svc_series, corr_thr=0.30):
    """Build directed topology from lag-1 cross-correlations between services.

    Vectorised: stacks all available service series into a matrix and
    computes the full n_svc × n_svc lag-1 correlation matrix in one
    numpy matmul instead of a nested Python loop.  ~100× faster for
    systems with many services.
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

    # Lag-1 correlation matrix: C[i,j] = corr(X[i], Y[j]) = X_i leads Y_j
    C = X @ Y.T  # (n_svcs, n_svcs)
    np.fill_diagonal(C, 0.)

    rows, cols = np.where(C > corr_thr)
    return [(svcs[i], svcs[j]) for i, j in zip(rows, cols)]


# ==========================================
# Shared core pipeline (steps 1–5)
# ==========================================


def _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs):
    """
    Shared pipeline: window split → preprocess → dim-reduction → GA-Banzhaf → topology.

    Returns a dict with all intermediate results needed by the entry points.
    """
    n_generations = int(kwargs.get("n_generations", 8))
    pop_size      = int(kwargs.get("pop_size", 20))
    ema_alpha     = float(kwargs.get("ema_alpha", 0.2))
    max_features  = int(kwargs.get("max_features", 75))
    rng           = np.random.default_rng(int(kwargs.get("seed", 42)))

    # ── 1. Window split ──────────────────────────────────────────────────────
    if anomalies is not None:
        sr = int(anomalies[0])
        normal_raw = data.head(sr)
        anomal_raw = data.tail(len(data) - sr)
    elif inject_time is not None and "time" in data.columns:
        normal_raw = data[data["time"] < inject_time]
        anomal_raw = data[data["time"] >= inject_time]
    else:
        sr = len(data) // 2
        normal_raw = data.head(sr)
        anomal_raw = data.tail(len(data) - sr)

    # ── 2. Preprocess jointly ────────────────────────────────────────────────
    dk = kwargs.get("dk_select_useful", False)
    n_rows = len(normal_raw)
    full_proc = preprocess(
        pd.concat([normal_raw, anomal_raw], ignore_index=True), dataset, dk
    )
    normal_df = full_proc.iloc[:n_rows].reset_index(drop=True)
    anomal_df = full_proc.iloc[n_rows:].reset_index(drop=True)

    shared = [c for c in normal_df.columns if c in anomal_df.columns]
    normal_df = normal_df[shared]
    anomal_df = anomal_df[shared]

    ml = min(len(normal_df), len(anomal_df))
    normal_df = normal_df.tail(ml).reset_index(drop=True)
    anomal_df = anomal_df.head(ml).reset_index(drop=True)

    all_cols = list(normal_df.columns)

    # ── 3. Anomaly pre-score + dimensionality reduction ──────────────────────
    full_df = pd.concat([normal_df, anomal_df], ignore_index=True)
    no_time = full_df.drop(columns=["time"], errors="ignore")

    _nm = normal_df.drop(columns=["time"], errors="ignore").reindex(columns=no_time.columns).values.astype(float)
    _am = anomal_df.drop(columns=["time"], errors="ignore").reindex(columns=no_time.columns).values.astype(float)

    if _nm.shape[1] > max_features * 2:
        _fast_mean = _nm.mean(axis=0)
        _fast_std  = _nm.std(axis=0)
        _fast_sc   = np.where(_fast_std > 0, _fast_std, 1.)
        _fast_anom = np.abs((_am - _fast_mean) / _fast_sc).max(axis=0)
        _top_idx   = np.argpartition(_fast_anom, -max_features * 2)[-max_features * 2:]
        _top_cols  = no_time.columns[_top_idx]
        _nm2 = _nm[:, _top_idx]
        _am2 = _am[:, _top_idx]
        _cols2 = _top_cols
    else:
        _nm2, _am2, _cols2 = _nm, _am, no_time.columns

    _med = np.nanmedian(_nm2, axis=0)
    _iqr = np.nanpercentile(_nm2, 75, axis=0) - np.nanpercentile(_nm2, 25, axis=0)
    _sc  = np.where(_iqr > 0, _iqr, 1.)
    _pre_anom_reduced = np.clip(np.abs((_am2 - _med) / _sc), 0., 100.).max(axis=0)

    if _nm.shape[1] > max_features * 2:
        _pre_anom = dict(zip(no_time.columns,
                             np.abs((_am - _nm.mean(axis=0)) /
                                    np.where(_nm.std(axis=0) > 0, _nm.std(axis=0), 1.)).max(axis=0)))
        for col, score in zip(_cols2, _pre_anom_reduced):
            _pre_anom[col] = float(score)
    else:
        _pre_anom = dict(zip(_cols2, _pre_anom_reduced))

    reduced = _redundancy_filtering(no_time, max_features=max_features, anom_scores=_pre_anom)

    # ── Force service coverage ───────────────────────────────────────────────
    covered_svcs = set(_parse_svc(c) for c in reduced.columns)
    svc_to_cols: dict = {}
    for c in no_time.columns:
        svc_to_cols.setdefault(_parse_svc(c), []).append(c)

    extra_cols = []
    for svc, cols in svc_to_cols.items():
        if svc not in covered_svcs:
            best = max(cols, key=lambda c: _pre_anom.get(c, 0.))
            extra_cols.append(best)
    if extra_cols:
        reduced = pd.concat([reduced, no_time[extra_cols]], axis=1)

    red_cols = list(reduced.columns)
    red_normal = reduced.values[:ml].astype(float)
    red_anomal = reduced.values[ml:].astype(float)

    # ── 4. GA-Banzhaf ────────────────────────────────────────────────────────
    banzhaf_w = _approximate_banzhaf_ga(
        red_normal, red_anomal,
        n_generations=n_generations,
        pop_size=pop_size,
        ema_alpha=ema_alpha,
        rng=rng,
    )

    metric_banzhaf = {red_cols[i]: float(banzhaf_w[i]) for i in range(len(red_cols))}

    svc_banzhaf: dict = {}
    for col, bw in metric_banzhaf.items():
        svc = _parse_svc(col)
        svc_banzhaf[svc] = max(svc_banzhaf.get(svc, 0.), bw)

    svc_anom: dict = {}
    for col, score in _pre_anom.items():
        svc = _parse_svc(col)
        svc_anom[svc] = max(svc_anom.get(svc, 0.), score)

    # ── 5. Topology (cross-correlation) ──────────────────────────────────────
    svc_names = sorted(set(_parse_svc(c) for c in all_cols))
    svc_series_all = {}
    for svc in svc_names:
        idxs = [i for i, c in enumerate(red_cols) if _parse_svc(c) == svc]
        if idxs:
            svc_series_all[svc] = reduced.values[:, idxs].max(axis=1).astype(float)
    topology_edges = _build_topology(svc_names, svc_series_all)

    return dict(
        all_cols=all_cols,
        metric_banzhaf=metric_banzhaf,
        svc_banzhaf=svc_banzhaf,
        svc_anom=svc_anom,
        svc_names=svc_names,
        topology_edges=topology_edges,
    )


def _rank_from_svc_scores(all_cols, svc_scores, metric_banzhaf, svc_anom=None):
    """Rank columns by combined service and metric scores.

    Weights:
      - svc_scores   : primary service-level signal  (0.5)
      - svc_anom     : anomaly score (fallback signal) (0.3) — when provided
      - metric_banzhaf: metric-level fine-grained signal (0.2 / 0.4)

    All signals are min-max normalised before combining so no single
    component dominates due to scale differences.
    """
    def _norm(d):
        vals = list(d.values())
        lo, hi = min(vals, default=0.), max(vals, default=1.)
        span = hi - lo if hi > lo else 1.
        return {k: (v - lo) / span for k, v in d.items()}

    norm_svc  = _norm(svc_scores)
    norm_mb   = _norm(metric_banzhaf)
    norm_anom = _norm(svc_anom) if svc_anom else {}

    def _col_score(col):
        svc = _parse_svc(col)
        bz  = norm_svc.get(svc, 0.)
        mb  = norm_mb.get(col, 0.)
        if norm_anom:
            an = norm_anom.get(svc, 0.)
            return bz * 0.5 + an * 0.3 + mb * 0.2
        return bz * 0.6 + mb * 0.4

    return sorted(all_cols, key=_col_score, reverse=True)


def _reverse_pagerank(svc_scores, topology_edges, alpha=0.85, max_iter=200,
                      svc_anom=None, anom_weight=0.3):
    """Personalized PageRank on the reversed causal graph.

    Root causes tend to sit upstream.  Running PPR on the *reversed*
    graph propagates probability mass toward nodes that can explain many
    downstream anomalies.

    Personalization vector combines Banzhaf scores and (optionally) anomaly
    scores so that high-signal services seed the random walk more strongly:
        p(s) ∝ (1 - anom_weight) × banzhaf_norm(s)
               + anom_weight      × anom_norm(s)
    """
    G = nx.DiGraph()
    G.add_nodes_from(svc_scores.keys())
    for u, v in topology_edges:
        if u in svc_scores and v in svc_scores:
            G.add_edge(v, u)          # reversed edge: downstream → upstream

    def _norm(d):
        lo, hi = min(d.values(), default=0.), max(d.values(), default=1.)
        span = hi - lo if hi > lo else 1.
        return {k: (v - lo) / span for k, v in d.items()}

    nb = _norm(svc_scores)
    if svc_anom:
        na = _norm(svc_anom)
        combined = {s: (1. - anom_weight) * nb.get(s, 0.)
                       + anom_weight       * na.get(s, 0.)
                    for s in svc_scores}
    else:
        combined = nb

    total = sum(combined.values()) or 1.
    personalization = {s: v / total for s, v in combined.items()}

    if G.number_of_edges() == 0:
        return dict(svc_scores)       # no topology → fall back to raw scores

    try:
        pr = nx.pagerank(G, alpha=alpha, personalization=personalization,
                         max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        return dict(svc_scores)
    return pr


# ==========================================
# RCAEval entry point
# ==========================================

def gabanzhaf(data, inject_time=None, dataset=None, num_loop=None,
              sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf + Reversed Personalized PageRank.

    After computing per-service Banzhaf scores, run PPR on the reversed
    causal topology so that upstream root causes receive higher probability
    mass than downstream victims.  The personalization vector blends
    Banzhaf scores (70%) with anomaly scores (30%) for richer seeding.
    """
    alpha       = float(kwargs.get("pr_alpha", 0.85))
    anom_weight = float(kwargs.get("pr_anom_weight", 0.3))
    core  = _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs)
    svc_scores = _reverse_pagerank(
        core["svc_banzhaf"], core["topology_edges"], alpha=alpha,
        svc_anom=core["svc_anom"], anom_weight=anom_weight,
    )
    ranks = _rank_from_svc_scores(
        core["all_cols"], svc_scores, core["metric_banzhaf"],
        svc_anom=core["svc_anom"]
    )
    return {"ranks": ranks, "node_names": core["all_cols"]}
