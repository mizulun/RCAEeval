import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
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


def _eval_fitness_svc(pop, svc_anom_arr, topo_matrix, alpha_topo, penalty_weight):
    """Vectorised fitness for service-level GA-Banzhaf.

    coverage     = Σ svc_anom[v]  for selected v
    topo_redund  = Σ svc_anom[v]  for selected v that has at least one
                   selected upstream u→v  (v's anomaly is explained by u,
                   so selecting v is redundant given u is already selected)
    sparsity     = |S| × penalty / n_svcs
    """
    n_svcs       = pop.shape[1]
    sel_size     = pop.sum(axis=1)                               # (P,)
    coverage     = pop @ svc_anom_arr                            # (P,)
    upstream_sel = pop @ topo_matrix                             # (P, S)
    explained    = (upstream_sel > 0).astype(np.float32) * pop  # (P, S)
    topo_redund  = explained @ svc_anom_arr                      # (P,)
    sparsity     = sel_size * penalty_weight / max(1, n_svcs)
    fitness      = coverage - alpha_topo * topo_redund - sparsity
    fitness[sel_size == 0] = 0.
    return fitness


def _approximate_banzhaf_ga_svc(svc_anom_arr, topo_matrix, high_sev_mask,
                                 n_generations=10, pop_size=20, ema_alpha=0.2,
                                 alpha_topo=0.3, rng=None):
    """Service-level GA-Banzhaf with topology-aware fitness.

    Binary vector over services: 1 = this service is part of the fault explanation.

    fitness = coverage − alpha_topo × topo_redundancy − sparsity

    Topology is baked into the fitness: downstream victims (explained by a
    selected upstream) are penalised, so their Banzhaf values converge lower
    than the true root cause without any separate PageRank step.

    GA population is initialised with a GMM-derived bias: high-severity
    services (from symptom clustering) have p=0.7, others p=0.3.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_svcs      = len(svc_anom_arr)
    ema_banzhaf = np.zeros(n_svcs)
    elite_k     = max(2, pop_size // 10)
    p_mut_base  = 1.0 / max(1, n_svcs)
    warmup_gens = n_generations // 3

    # GMM-biased initialisation
    p_init = np.where(high_sev_mask > 0, 0.7, 0.3)
    pop    = (rng.random((pop_size, n_svcs)) < p_init).astype(np.float32)
    empty  = pop.sum(axis=1) == 0
    if empty.any():
        fix_idx = rng.integers(n_svcs, size=empty.sum())
        pop[np.where(empty)[0], fix_idx] = 1.0

    for gen in range(n_generations):
        penalty_weight = min(0.1, 0.05 * (1.0 + gen / n_generations))

        fitness = _eval_fitness_svc(pop, svc_anom_arr, topo_matrix,
                                    alpha_topo, penalty_weight)

        # ── Banzhaf marginal (vectorised over all services) ──────────────────
        w_count  = pop.sum(axis=0)
        wo_count = pop_size - w_count
        w_sum    = (pop * fitness[:, None]).sum(axis=0)
        wo_sum   = ((1. - pop) * fitness[:, None]).sum(axis=0)
        w_mean   = np.where(w_count  > 0, w_sum  / np.maximum(w_count,  1.), 0.)
        wo_mean  = np.where(wo_count > 0, wo_sum / np.maximum(wo_count, 1.), 0.)
        current  = np.maximum(0., w_mean - wo_mean)
        ema_banzhaf = ema_alpha * current + (1 - ema_alpha) * ema_banzhaf

        # ── Elitism ──────────────────────────────────────────────────────────
        elite_idx = np.argsort(fitness)[-elite_k:]
        next_pop  = list(pop[elite_idx])

        # ── Guided mutation (high-Banzhaf services mutate less) ──────────────
        if gen >= warmup_gens and ema_banzhaf.max() > 0:
            b_norm = ((ema_banzhaf - ema_banzhaf.min()) /
                      (ema_banzhaf.max() - ema_banzhaf.min() + 1e-9))
            p_flip = p_mut_base * (1.0 + (1.0 - b_norm))
        else:
            p_flip = np.full(n_svcs, p_mut_base * 2)

        # ── Batch crossover + mutation ────────────────────────────────────────
        n_offspring = pop_size - elite_k
        p1_idx   = rng.integers(0, pop_size, size=n_offspring)
        p2_idx   = rng.integers(0, pop_size, size=n_offspring)
        mask     = rng.random((n_offspring, n_svcs)) < 0.5
        children = np.where(mask, pop[p1_idx], pop[p2_idx]).copy()
        flip     = rng.random((n_offspring, n_svcs)) < p_flip
        children[flip] = 1.0 - children[flip]
        empty    = children.sum(axis=1) == 0
        if empty.any():
            fix_idx = rng.integers(n_svcs, size=empty.sum())
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
# Core pipeline
# ==========================================


def _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs):
    """
    Pipeline: window split → z-score → topology → pre-filter → GMM bias
              → service-level GA-Banzhaf (topology in fitness) → rank
    """
    n_generations = int(kwargs.get("n_generations", 10))
    pop_size      = int(kwargs.get("pop_size", 20))
    ema_alpha     = float(kwargs.get("ema_alpha", 0.2))
    top_k_svcs    = int(kwargs.get("top_k_svcs", 15))
    rng           = np.random.default_rng(int(kwargs.get("seed", 42)))
    trace_scores  = kwargs.get("trace_scores", None)

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

    shared    = [c for c in normal_df.columns if c in anomal_df.columns]
    normal_df = normal_df[shared]
    anomal_df = anomal_df[shared]

    ml        = min(len(normal_df), len(anomal_df))
    normal_df = normal_df.tail(ml).reset_index(drop=True)
    anomal_df = anomal_df.head(ml).reset_index(drop=True)

    all_cols = list(normal_df.columns)
    no_time  = pd.concat([normal_df, anomal_df], ignore_index=True).drop(columns=["time"], errors="ignore")
    cols     = no_time.columns

    _nm = normal_df.drop(columns=["time"], errors="ignore").values.astype(float)
    _am = anomal_df.drop(columns=["time"], errors="ignore").values.astype(float)

    # ── 3. Per-metric anomaly score (IQR z-score) ────────────────────────────
    _med = np.nanmedian(_nm, axis=0)
    _iqr = np.nanpercentile(_nm, 75, axis=0) - np.nanpercentile(_nm, 25, axis=0)
    _sc  = np.where(_iqr > 0, _iqr, 1.)

    def _norm01(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-9)

    anom_scores = _norm01(np.nanmax(np.abs((_am - _med) / _sc), axis=0))
    _pre_anom   = dict(zip(cols, anom_scores))

    # Per-service: take max across all its metrics
    svc_anom_dict: dict = {}
    for col, score in _pre_anom.items():
        svc = _parse_svc(col)
        svc_anom_dict[svc] = max(svc_anom_dict.get(svc, 0.), score)

    # ── 4. Build topology (adaptive corr_thr) ────────────────────────────────
    svc_names = sorted(set(_parse_svc(c) for c in all_cols))
    svc_series_full = {}
    for svc in svc_names:
        idxs = [i for i, c in enumerate(cols) if _parse_svc(c) == svc]
        if idxs:
            svc_series_full[svc] = no_time.values[:, idxs].max(axis=1).astype(float)

    n_svcs_count = len(svc_names)
    corr_thr = 0.30 if n_svcs_count <= 10 else (0.40 if n_svcs_count <= 20 else 0.50)
    topology_edges = _build_topology(svc_names, svc_series_full, corr_thr=corr_thr)

    upstream_map: dict = {s: set() for s in svc_names}
    for u, v in topology_edges:
        upstream_map.get(v, set()).add(u)

    # ── 5. Pre-filter: top-K services by svc_anom + upstream rescue ──────────
    svc_ranked     = sorted(svc_names, key=lambda s: svc_anom_dict.get(s, 0.), reverse=True)
    candidate_svcs = set(svc_ranked[:top_k_svcs])
    for svc in list(candidate_svcs):           # upstream rescue for LOSS/DELAY
        candidate_svcs |= upstream_map.get(svc, set())

    cand_list = sorted(candidate_svcs)
    n_cand    = len(cand_list)
    svc_idx   = {s: i for i, s in enumerate(cand_list)}

    # ── 6. GMM symptom clustering → high-severity mask ───────────────────────
    svc_cluster_map, cluster_severities = _symptom_cluster(cols, anom_scores)
    sorted_clusters = sorted(cluster_severities.keys(),
                             key=lambda k: cluster_severities[k], reverse=True)
    top_clusters  = sorted_clusters[:2] if len(sorted_clusters) > 2 else sorted_clusters[:1]
    high_sev_svcs = {svc for svc, c_id in svc_cluster_map.items() if c_id in top_clusters}
    high_sev_mask = np.array([1. if s in high_sev_svcs else 0. for s in cand_list])

    # ── 7. Service-level topology matrix + anomaly array (candidates only) ───
    svc_anom_arr = np.array([svc_anom_dict.get(s, 0.) for s in cand_list])
    topo_matrix  = np.zeros((n_cand, n_cand), dtype=np.float32)
    for u, v in topology_edges:
        if u in svc_idx and v in svc_idx:
            topo_matrix[svc_idx[u], svc_idx[v]] = 1.

    # ── 8. Service-level GA-Banzhaf ───────────────────────────────────────────
    ema_banzhaf = _approximate_banzhaf_ga_svc(
        svc_anom_arr, topo_matrix, high_sev_mask,
        n_generations=n_generations,
        pop_size=pop_size,
        ema_alpha=ema_alpha,
        rng=rng,
    )
    svc_banzhaf = {cand_list[i]: float(ema_banzhaf[i]) for i in range(n_cand)}

    # ── 9. Final ranking ──────────────────────────────────────────────────────
    # Signals: Banzhaf (causal contribution) + svc_anom (direct signal)
    #          + trace (error rate / duration ratio, if available)
    def _norm(d):
        lo, hi = min(d.values(), default=0.), max(d.values(), default=1.)
        span = hi - lo if hi > lo else 1.
        return {k: (v - lo) / span for k, v in d.items()}

    nb = _norm(svc_banzhaf)
    na = _norm(svc_anom_dict)

    if trace_scores:
        nt = _norm({s: trace_scores.get(s, 0.) for s in svc_anom_dict})
        def _col_score(col):
            svc = _parse_svc(col)
            return nb.get(svc, 0.) * 0.60 + na.get(svc, 0.) * 0.25 + nt.get(svc, 0.) * 0.15
    else:
        def _col_score(col):
            svc = _parse_svc(col)
            return nb.get(svc, 0.) * 0.70 + na.get(svc, 0.) * 0.30

    ranks = sorted(all_cols, key=_col_score, reverse=True)
    return {"ranks": ranks, "node_names": all_cols}


# ==========================================
# RCAEval entry point
# ==========================================

def gabanzhaf(data, inject_time=None, dataset=None, num_loop=None,
              sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf with topology-aware fitness + multi-modal ranking.

    Service-level GA approximates Banzhaf power indices where the fitness
    function directly penalises selecting downstream victims (causal
    redundancy via topology), so no separate PageRank step is needed.

    Trace signals (error rate / duration ratio) are fused into the final
    ranking when traces.csv is available.
    """
    args_ns  = kwargs.get("args")
    case_dir = (os.path.dirname(args_ns.data_path)
                if args_ns and hasattr(args_ns, "data_path") else None)
    _t_inject    = inject_time if inject_time is not None else 0
    trace_scores = _trace_onset_scores(case_dir, _t_inject)

    kwargs["trace_scores"] = trace_scores
    return _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs)
