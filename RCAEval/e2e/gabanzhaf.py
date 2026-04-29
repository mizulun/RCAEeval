import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

from RCAEval.io.time_series import preprocess


def _trace_onset_scores(case_dir, inject_time):
    """Lightweight trace-based per-service anomaly signal.

    Reads only 4 columns (serviceName, startTimeMillis, statusCode, duration)
    in chunks with an early-stop cap so total I/O stays < 0.5 s.

    Returns {svc_name: score ∈ [0, 1]} or None if traces unavailable.

    Signals:
      - Error rate  (covers LOSS): fraction of spans with statusCode ≠ 0
      - Duration ratio (covers DELAY): anomaly-period avg-duration /
                                        normal-period avg-duration − 1,
                                        capped at 10× and normalised to [0,1]
    Per-service score = max(error_rate, duration_signal).
    """
    if not case_dir:
        return None
    tp = os.path.join(case_dir, "traces.csv")
    if not os.path.exists(tp):
        return None

    try:
        inject_ms   = inject_time * 1000
        needed      = ["serviceName", "startTimeMillis", "statusCode", "duration"]
        MAX_ANOMAL  = 30_000        # row cap to bound runtime

        chunks_normal: list = []
        chunks_anomal: list = []
        anomal_rows = 0

        for chunk in pd.read_csv(tp, usecols=needed, chunksize=10_000,
                                  on_bad_lines="skip"):
            ts       = pd.to_numeric(chunk["startTimeMillis"], errors="coerce")
            is_norm  = ts < inject_ms
            is_anom  = ts >= inject_ms

            if is_norm.any():
                chunks_normal.append(chunk[is_norm])
            if is_anom.any():
                ac = chunk[is_anom]
                chunks_anomal.append(ac)
                anomal_rows += len(ac)

            if anomal_rows >= MAX_ANOMAL:
                break

        if not chunks_anomal:
            return None

        anomal_df = pd.concat(chunks_anomal, ignore_index=True)
        anomal_df["is_err"] = (
            pd.to_numeric(anomal_df["statusCode"], errors="coerce").fillna(0.) != 0.
        )
        anomal_df["dur"] = pd.to_numeric(anomal_df["duration"], errors="coerce").fillna(0.)

        a_stats = anomal_df.groupby("serviceName", sort=False).agg(
            cnt    =("serviceName", "size"),
            err_cnt=("is_err",      "sum"),
            dur_sum=("dur",         "sum"),
        )

        # Normal-period duration baseline
        n_dur_avg: dict = {}
        if chunks_normal:
            normal_df = pd.concat(chunks_normal, ignore_index=True)
            normal_df["dur"] = pd.to_numeric(normal_df["duration"], errors="coerce").fillna(0.)
            n_stats = normal_df.groupby("serviceName", sort=False).agg(
                cnt    =("serviceName", "size"),
                dur_sum=("dur",         "sum"),
            )
            for svc, row in n_stats.iterrows():
                n_dur_avg[str(svc)] = float(row["dur_sum"]) / max(int(row["cnt"]), 1)

        scores: dict = {}
        for svc, row in a_stats.iterrows():
            svc      = str(svc)
            a_cnt    = max(int(row["cnt"]), 1)
            err_rate = float(row["err_cnt"]) / a_cnt

            a_avg_dur = float(row["dur_sum"]) / a_cnt
            n_avg_dur = n_dur_avg.get(svc, 1.)
            dur_ratio = (a_avg_dur / max(n_avg_dur, 1.)) - 1.
            dur_signal = max(0., min(dur_ratio, 10.)) / 10.

            scores[svc] = max(err_rate, dur_signal)

        if not scores:
            return None

        lo, hi = min(scores.values()), max(scores.values())
        span = hi - lo if hi > lo else 1.
        return {svc: (v - lo) / span for svc, v in scores.items()}

    except Exception:
        return None


def _symptom_cluster(cols, anom_scores):
    """
    TORAI SymptomCluster 實作：
    為每個服務建立嚴重度向量，並使用 GMM 進行聚類，找出異常行為相似的服務群組。
    """
    svc_to_scores = {}
    for col, score in zip(cols, anom_scores):
        svc = _parse_svc(col)
        svc_to_scores.setdefault(svc, []).append(score)

    svcs = list(svc_to_scores.keys())
    severity_vectors = []

    for svc in svcs:
        scores = np.array(svc_to_scores[svc])
        severity_vectors.append([
            np.max(scores),
            np.mean(scores),
            np.percentile(scores, 90) if len(scores) > 1 else np.max(scores)
        ])

    X = np.array(severity_vectors)
    n_components = min(4, len(svcs))

    if n_components < 2:
        return {svcs[0]: 0}, {0: float(np.mean(X[:, 0]))}

    best_gmm = None
    best_bic = np.inf
    for k in range(1, n_components + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-3)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except ValueError:
            continue

    if best_gmm is None:
        return {svc: 0 for svc in svcs}, {0: float(X[:, 0].mean())}

    labels = best_gmm.predict(X)
    cluster_severities = {k: float(np.mean(X[labels == k][:, 0])) for k in np.unique(labels)}
    svc_cluster_map = {svcs[i]: labels[i] for i in range(len(svcs))}

    return svc_cluster_map, cluster_severities


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
    Shared pipeline: window split → robust score → symptom clustering → dim-reduction → GA-Banzhaf → topology.
    """
    n_generations = int(kwargs.get("n_generations", 10))
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
    full_proc = preprocess(pd.concat([normal_raw, anomal_raw], ignore_index=True), dataset, dk)
    
    normal_df = full_proc.iloc[:n_rows].reset_index(drop=True)
    anomal_df = full_proc.iloc[n_rows:].reset_index(drop=True)

    shared = [c for c in normal_df.columns if c in anomal_df.columns]
    normal_df = normal_df[shared]
    anomal_df = anomal_df[shared]

    ml = min(len(normal_df), len(anomal_df))
    normal_df = normal_df.tail(ml).reset_index(drop=True)
    anomal_df = anomal_df.head(ml).reset_index(drop=True)
    
    all_cols = list(normal_df.columns)
    no_time = pd.concat([normal_df, anomal_df], ignore_index=True).drop(columns=["time"], errors="ignore")
    cols = no_time.columns

    _nm = normal_df.drop(columns=["time"], errors="ignore").values.astype(float)
    _am = anomal_df.drop(columns=["time"], errors="ignore").values.astype(float)

    # ── 3. 複合異常分數：z-score + onset time + variance change ─────────────
    _med = np.nanmedian(_nm, axis=0)
    _iqr = np.nanpercentile(_nm, 75, axis=0) - np.nanpercentile(_nm, 25, axis=0)
    _sc  = np.where(_iqr > 0, _iqr, 1.)
    z_am = np.abs((_am - _med) / _sc)

    # Signal A: robust z-score (原始 severity)
    sig_zscore = np.nanmax(z_am, axis=0)

    # GMM 和 redundancy filter 只用 robust z-score（正規化到 [0,1]）
    # onset/variance 對非 LOSS 故障有副作用，不混入 pre_anom
    def _norm01(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-9)

    anom_scores = _norm01(sig_zscore)   # GMM 需要 [0,1] scale
    _pre_anom   = dict(zip(cols, anom_scores))

    # ── 4. 提前建拓樸 (用全部服務的資料) ────────────────────────────────────
    # 在 GMM 篩選之前先建好，這樣才能把 target_svcs 的上游服務也拉進來，
    # 避免 LOSS 根因（upstream、信號弱）在 GMM 階段就被排除。
    svc_names = sorted(set(_parse_svc(c) for c in all_cols))
    svc_series_full = {}
    for svc in svc_names:
        svc_col_idxs = [i for i, c in enumerate(cols) if _parse_svc(c) == svc]
        if svc_col_idxs:
            svc_series_full[svc] = no_time.values[:, svc_col_idxs].max(axis=1).astype(float)
    # Adaptive corr_thr: 服務越多、假邊越多，用更嚴格的閾值讓拓樸更稀疏可靠
    n_svcs_count = len(svc_names)
    corr_thr = 0.30 if n_svcs_count <= 10 else (0.40 if n_svcs_count <= 20 else 0.50)
    topology_edges = _build_topology(svc_names, svc_series_full, corr_thr=corr_thr)

    # upstream_map[v] = {u | u→v 存在於拓樸} (即 v 的直接上游)
    upstream_map: dict = {s: set() for s in svc_names}
    for u, v in topology_edges:
        upstream_map.get(v, set()).add(u)

    # ── 5. TORAI Symptom Clustering (GMM 分群) ──────────────────────────────
    svc_cluster_map, cluster_severities = _symptom_cluster(cols, anom_scores)

    sorted_clusters = sorted(cluster_severities.keys(), key=lambda k: cluster_severities[k], reverse=True)
    top_clusters = sorted_clusters[:2] if len(sorted_clusters) > 2 else sorted_clusters[:1]
    target_svcs = {svc for svc, c_id in svc_cluster_map.items() if c_id in top_clusters}

    # ── LOSS 補救：把 target_svcs 的直接上游也加入 ────────────────────────────
    # LOSS 根因 A 的 metric 信號弱 → GMM 不會把 A 選進 top cluster。
    # 但 A→B/C/D 的邊在拓樸裡存在，所以把 B/C/D 的上游（即 A）補進來。
    upstream_of_targets = set()
    for svc in target_svcs:
        upstream_of_targets |= upstream_map.get(svc, set())
    target_svcs |= upstream_of_targets

    # 只保留目標服務的特徵進行後續運算
    # 若 target_svcs 為空（極端情況）則退回全部欄位
    target_cols = [c for c in cols if _parse_svc(c) in target_svcs]
    no_time_filtered = no_time[target_cols] if target_cols else no_time

    # ── 6. Dimensionality Reduction (針對高危險群 + 上游服務) ────────────────
    reduced = _redundancy_filtering(no_time_filtered, max_features=max_features, anom_scores=_pre_anom)

    # 確保每個 target 服務至少有一個代表性指標
    covered_svcs = set(_parse_svc(c) for c in reduced.columns)
    extra_cols = []
    for svc in target_svcs:
        if svc not in covered_svcs:
            svc_cols = [c for c in cols if _parse_svc(c) == svc]
            if svc_cols:
                best = max(svc_cols, key=lambda c: _pre_anom.get(c, 0.))
                extra_cols.append(best)

    if extra_cols:
        reduced = pd.concat([reduced, no_time[extra_cols]], axis=1)

    red_cols = list(reduced.columns)
    red_normal = reduced.values[:ml].astype(float)
    red_anomal = reduced.values[ml:].astype(float)

    # ── 7. GA-Banzhaf ────────────────────────────────────────────────────────
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

    # ── 方向 1: Residual Anomaly ──────────────────────────────────────────────
    # svc_residual[s] = svc_anom[s] − max(upstream anomaly)
    # 若上游 u 的異常已經很高，s 的異常可被解釋為傳播效應（受害者）。
    # 無法被上游解釋的殘差高 → 更可能是根因。
    svc_residual: dict = {}
    for svc, score in svc_anom.items():
        upstream_max = max(
            (svc_anom.get(u, 0.) for u in upstream_map.get(svc, set())),
            default=0.
        )
        svc_residual[svc] = max(0., score - upstream_max)

    # ── 方向 4: Upstream Hub Score ────────────────────────────────────────────
    # hub[s] = Σ svc_anom[v] for all v downstream of s
    # 同時是多個高異常服務的上游 → 更可能是共同根因
    svc_hub: dict = {s: 0. for s in svc_names}
    for u, v in topology_edges:
        if u in svc_hub:
            svc_hub[u] += svc_anom.get(v, 0.)

    return dict(
        all_cols=all_cols,
        metric_banzhaf=metric_banzhaf,
        svc_banzhaf=svc_banzhaf,
        svc_anom=svc_anom,
        svc_residual=svc_residual,
        svc_hub=svc_hub,
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
                      svc_anom=None, svc_residual=None, svc_hub=None,
                      trace_scores=None, anom_weight=0.3):
    """Personalized PageRank on the reversed causal graph.

    Personalization vector blends five signals:
      - banzhaf   (0.55/0.60): GA-Banzhaf marginal contribution
      - residual  (0.20):      unexplained anomaly after subtracting upstream influence
      - hub       (0.10):      sum of downstream anomaly scores
      - trace     (0.10):      trace-based anomaly (error rate / duration ratio)
                               only when trace_scores is provided, else weight goes to banzhaf
      - anom      (0.05):      raw anomaly score (fallback signal)
    """
    G = nx.DiGraph()
    G.add_nodes_from(svc_scores.keys())
    for u, v in topology_edges:
        if u in svc_scores and v in svc_scores:
            G.add_edge(v, u)          # reversed: downstream → upstream

    def _norm(d):
        lo, hi = min(d.values(), default=0.), max(d.values(), default=1.)
        span = hi - lo if hi > lo else 1.
        return {k: (v - lo) / span for k, v in d.items()}

    nb  = _norm(svc_scores)
    nr  = _norm(svc_residual) if svc_residual else nb
    nh  = _norm(svc_hub)      if svc_hub      else {s: 0. for s in svc_scores}
    na  = _norm(svc_anom)     if svc_anom     else {s: 0. for s in svc_scores}

    # Adaptive weights: 服務少時 banzhaf 主導（拓樸簡單，直接信號可靠）
    #                   服務多時加重 residual/hub（拓樸複雜，需要因果定位）
    n_svcs = len(svc_scores)
    if n_svcs <= 10:
        w_bz, w_nr, w_nh = 0.65, 0.20, 0.10
    elif n_svcs <= 20:
        w_bz, w_nr, w_nh = 0.55, 0.25, 0.15
    else:
        w_bz, w_nr, w_nh = 0.45, 0.30, 0.20

    if trace_scores is not None:
        # trace 權重從 banzhaf 撥出 0.10
        w_bz_t = max(w_bz - 0.10, 0.35)
        ts_proj = {s: trace_scores.get(s, 0.) for s in svc_scores}
        lo, hi  = min(ts_proj.values()), max(ts_proj.values())
        span    = hi - lo if hi > lo else 1.
        nt      = {s: (v - lo) / span for s, v in ts_proj.items()}

        combined = {
            s: w_bz_t * nb.get(s, 0.)
             + w_nr   * nr.get(s, 0.)
             + w_nh   * nh.get(s, 0.)
             + 0.10   * nt.get(s, 0.)
             + 0.05   * na.get(s, 0.)
            for s in svc_scores
        }
    else:
        combined = {
            s: w_bz * nb.get(s, 0.)
             + w_nr  * nr.get(s, 0.)
             + w_nh  * nh.get(s, 0.)
             + 0.05  * na.get(s, 0.)
            for s in svc_scores
        }

    total = sum(combined.values()) or 1.
    personalization = {s: v / total for s, v in combined.items()}

    if G.number_of_edges() == 0:
        return dict(svc_scores)

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

    # Lightweight trace onset signal (helps DELAY / LOSS differentiation)
    args_ns  = kwargs.get("args")
    case_dir = (os.path.dirname(args_ns.data_path)
                if args_ns and hasattr(args_ns, "data_path") else None)
    _t_inject = inject_time if inject_time is not None else 0
    trace_scores = _trace_onset_scores(case_dir, _t_inject)

    core  = _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs)
    svc_scores = _reverse_pagerank(
        core["svc_banzhaf"], core["topology_edges"], alpha=alpha,
        svc_anom=core["svc_anom"],
        svc_residual=core["svc_residual"],
        svc_hub=core["svc_hub"],
        trace_scores=trace_scores,
        anom_weight=anom_weight,
    )
    ranks = _rank_from_svc_scores(
        core["all_cols"], svc_scores, core["metric_banzhaf"],
        svc_anom=core["svc_anom"]
    )
    return {"ranks": ranks, "node_names": core["all_cols"]}
