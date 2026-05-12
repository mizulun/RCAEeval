import os
import numpy as np
import pandas as pd
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


def _extract_call_graph(traces_path):
    """Extract service call graph from the first 50 K trace spans.

    Reads only spanID / parentSpanID / serviceName, so I/O is ~0.15 s even
    for the largest TT files (114 MB).  50 K rows covers ≥ 98 % of distinct
    call pairs in practice.

    Returns
    -------
    dict  {caller_svc: set(callee_svc, ...)}
        "caller calls callee" — i.e. if callee is anomalous it *explains*
        caller's symptoms (anomaly propagates caller ← callee).
    """
    if not traces_path or not os.path.exists(traces_path):
        return {}
    try:
        df = pd.read_csv(
            traces_path,
            usecols=["serviceName", "spanID", "parentSpanID"],
            nrows=50_000,
            on_bad_lines="skip",
        )
        span_to_svc = df.set_index("spanID")["serviceName"].to_dict()
        df["parentSvc"] = df["parentSpanID"].map(span_to_svc)
        edges = df[
            df["parentSvc"].notna() & (df["parentSvc"] != df["serviceName"])
        ]
        result: dict = {}
        for caller, callee in zip(edges["parentSvc"], edges["serviceName"]):
            result.setdefault(caller, set()).add(callee)
        return result
    except Exception:
        return {}


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


def _metric_evidence(normal_mat, anomal_mat, cols):
    """Metric-level evidence from robust magnitude, persistence and peak.

    The score remains unsupervised: it only compares the post-injection window
    with the pre-injection baseline.  Percentile and persistence terms reduce
    the chance that a single noisy spike dominates the ranking.
    """
    med = np.nanmedian(normal_mat, axis=0)
    q75 = np.nanpercentile(normal_mat, 75, axis=0)
    q25 = np.nanpercentile(normal_mat, 25, axis=0)
    std = np.nanstd(normal_mat, axis=0)
    iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)

    rz = np.abs((anomal_mat - med) / iqr)
    rz = np.nan_to_num(rz, nan=0., posinf=0., neginf=0.)
    # Selective Winsorization: cap metrics where IQR=0 AND std≈0 (truly constant
    # baseline).  These produce inflated rz through the fallback denominator of
    # 1.0 even for tiny shifts, masking real root causes.  The std<=1e-6 guard
    # preserves columns that have IQR=0 but non-trivial spread (step-function
    # or bursty behaviour) so their rz is still meaningful.
    near_zero_baseline = ((q75 - q25) <= 0) & (std <= 1e-6)   # shape (n_cols,)
    rz = np.where(near_zero_baseline, np.clip(rz, 0., 10.), rz)

    mag = _norm01_arr(np.nanpercentile(rz, 95, axis=0))
    peak = _norm01_arr(np.nanmax(rz, axis=0))
    persistence = _norm01_arr(np.mean(rz > 3.0, axis=0))
    raw = 0.55 * mag + 0.30 * persistence + 0.15 * peak
    score = _norm01_arr(raw)

    return {
        col: {
            "score": float(score[i]),
            "mag": float(mag[i]),
            "persistence": float(persistence[i]),
            "peak": float(peak[i]),
        }
        for i, col in enumerate(cols)
    }


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
        arr = np.asarray([metric_ev[c]["score"] for c in svc_cols], dtype=float)
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
        sc    = metric_ev[col]["score"]

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
    upstream_sel = np.clip(pop @ topo_matrix, 0., 1.)            # (P, S)
    explained    = upstream_sel * pop                            # (P, S)
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

    GA population is initialised with a soft high-evidence bias: services with
    stronger unsupervised evidence are sampled more often in the initial pool.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_svcs      = len(svc_anom_arr)
    ema_banzhaf = np.zeros(n_svcs)
    elite_k     = max(2, pop_size // 10)
    p_mut_base  = 1.0 / max(1, n_svcs)
    warmup_gens = n_generations // 3

    # Evidence-biased initialisation
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


# ==========================================
# Core pipeline
# ==========================================


def _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs):
    """
    Pipeline: window split → metric evidence → service evidence
              → bounded weighted topology → GA-Banzhaf → metric ranking
    """
    n_generations = int(kwargs.get("n_generations", 10))
    pop_size      = int(kwargs.get("pop_size", 20))
    ema_alpha     = float(kwargs.get("ema_alpha", 0.2))
    top_k_svcs    = int(kwargs.get("top_k_svcs", 15))
    max_cand      = int(kwargs.get("max_candidate_svcs", top_k_svcs + 5))
    rng           = np.random.default_rng(int(kwargs.get("seed", 42)))
    trace_scores  = kwargs.get("trace_scores", None)
    sli           = kwargs.get("sli", None)

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

    no_time  = pd.concat([normal_df, anomal_df], ignore_index=True).drop(columns=["time"], errors="ignore")
    cols     = list(no_time.columns)
    all_cols = cols

    _nm = normal_df.drop(columns=["time"], errors="ignore").values.astype(float)
    _am = anomal_df.drop(columns=["time"], errors="ignore").values.astype(float)

    # ── 3. Metric and service evidence ───────────────────────────────────────
    metric_ev = _metric_evidence(_nm, _am, cols)
    metric_score = {c: metric_ev[c]["score"] for c in cols}
    svc_anom_dict = _service_evidence(cols, metric_ev)
    svc_names = sorted(svc_anom_dict.keys())

    # ── 4. Candidate-first bounded topology ──────────────────────────────────
    svc_ranked = sorted(svc_names, key=lambda s: svc_anom_dict.get(s, 0.), reverse=True)
    seed_svcs = set(svc_ranked[:top_k_svcs])
    if sli:
        seed_svcs.add(_parse_svc(str(sli)))

    pool_k = min(len(svc_ranked), max(top_k_svcs * 2, top_k_svcs + 5))
    pool_svcs = set(svc_ranked[:pool_k]) | seed_svcs

    svc_series_full = {}
    for svc in pool_svcs:
        idxs = [i for i, c in enumerate(cols) if _parse_svc(c) == svc]
        if idxs:
            svc_series_full[svc] = no_time.values[:, idxs].max(axis=1).astype(float)

    lag_corr_edges = _build_topology(sorted(pool_svcs), svc_series_full)

    # ── 4b. Augment topology with trace-derived call graph ────────────────────
    # Call graph: {caller: {callee}}.  When callee is anomalous it *explains*
    # caller's symptoms → topology edge: callee → caller (callee upstream of caller).
    # Confidence 0.5: moderate confidence so call graph edges don't dominate
    # topo_rel (which is computed from lag-corr edges only, keeping its
    # statistical meaning intact).  Guards: both callee AND caller must show
    # meaningful anomaly to avoid adding spurious edges in non-propagating faults.
    call_graph = kwargs.get("call_graph", {})
    call_graph_edges = []
    if call_graph:
        existing_edges = {(u, v) for u, v, _ in lag_corr_edges}
        for caller, callees in call_graph.items():
            for callee in callees:
                if callee not in pool_svcs or caller not in pool_svcs:
                    continue
                callee_score = svc_anom_dict.get(callee, 0.)
                caller_score = svc_anom_dict.get(caller, 0.)
                if callee_score < 0.2 or caller_score < 0.1:
                    continue
                if (callee, caller) not in existing_edges:
                    call_graph_edges.append((callee, caller, 0.5))
                    existing_edges.add((callee, caller))

    topology_edges = lag_corr_edges + call_graph_edges

    upstream_map: dict = {s: set() for s in svc_names}
    edge_conf: dict = {}
    for u, v, w in topology_edges:
        upstream_map.get(v, set()).add(u)
        edge_conf[(u, v)] = w

    # ── 5. Pre-filter with upstream rescue, then bound runtime ───────────────
    candidate_svcs = set(seed_svcs)
    for svc in list(candidate_svcs):
        candidate_svcs |= upstream_map.get(svc, set())
    if len(candidate_svcs) > max_cand:
        candidate_svcs = set(sorted(candidate_svcs,
                                    key=lambda s: svc_anom_dict.get(s, 0.),
                                    reverse=True)[:max_cand])

    cand_list = sorted(candidate_svcs)
    n_cand    = len(cand_list)
    svc_idx   = {s: i for i, s in enumerate(cand_list)}

    # ── 6. Soft high-evidence mask for GA initialisation ─────────────────────
    cand_anom_vals = np.array([svc_anom_dict.get(s, 0.) for s in cand_list])
    high_cut = np.percentile(cand_anom_vals, 60) if len(cand_anom_vals) else 1.
    high_sev_mask = np.array([1. if svc_anom_dict.get(s, 0.) >= high_cut else 0.
                              for s in cand_list])

    # ── 7. Weighted residual + topology matrix ───────────────────────────────
    svc_residual: dict = {}
    for svc, score in svc_anom_dict.items():
        explained = max(
            (svc_anom_dict.get(u, 0.) * edge_conf.get((u, svc), 0.)
             for u in upstream_map.get(svc, set())),
            default=0.
        )
        svc_residual[svc] = max(0., score - explained)

    svc_anom_arr     = np.array([svc_anom_dict.get(s, 0.)  for s in cand_list])
    svc_residual_arr = np.array([svc_residual.get(s, 0.)   for s in cand_list])

    # topo_rel is computed from lag-correlation edges only so call-graph edges
    # (different confidence scale) don't distort the residual weight.
    conf_vals = np.array([w for _, _, w in lag_corr_edges], dtype=float)
    topo_rel = float(np.clip(np.mean(conf_vals) * 2.0, 0.0, 0.65)) if conf_vals.size else 0.0
    res_w = 0.30 + 0.50 * topo_rel
    fitness_signal = res_w * svc_residual_arr + (1.0 - res_w) * svc_anom_arr

    topo_matrix = np.zeros((n_cand, n_cand), dtype=np.float32)
    for u, v, w in topology_edges:
        if u in svc_idx and v in svc_idx:
            topo_matrix[svc_idx[u], svc_idx[v]] = w

    # ── 8. Bounded GA-Banzhaf computation ────────────────────────────────────
    banzhaf_w = _approximate_banzhaf_ga_svc(
        fitness_signal, topo_matrix, high_sev_mask,
        n_generations=n_generations,
        pop_size=pop_size,
        ema_alpha=ema_alpha,
        rng=rng,
    )
    svc_banzhaf = {cand_list[i]: float(banzhaf_w[i]) for i in range(n_cand)}

    # ── 9. Final metric-level ranking ────────────────────────────────────────
    nb = _norm_dict(svc_banzhaf)
    nr = _norm_dict(svc_residual)
    na = _norm_dict(svc_anom_dict)
    nm = _norm_dict(metric_score)

    if trace_scores:
        trace_cov = sum(1 for s in svc_anom_dict if s in trace_scores) / max(1, len(svc_anom_dict))
        trace_rel = float(np.clip(trace_cov, 0.0, 0.25))
        nt = _norm_dict({s: trace_scores.get(s, 0.) for s in svc_anom_dict})
    else:
        trace_rel = 0.0
        nt = {}

    def _col_score(col):
        svc = _parse_svc(col)
        service_root = (
            0.45 * nb.get(svc, 0.)
            + 0.30 * nr.get(svc, 0.)
            + 0.25 * na.get(svc, 0.)
        )
        if trace_rel > 0:
            service_root = (1.0 - trace_rel) * service_root + trace_rel * nt.get(svc, 0.)

        metric_local = nm.get(col, 0.)
        # Keep service causality primary, but let metric evidence decide the
        # channel order inside a correctly localised service.
        return 0.70 * service_root + 0.30 * metric_local + 0.20 * service_root * metric_local

    ranks = sorted(all_cols, key=_col_score, reverse=True)
    return {"ranks": ranks, "node_names": all_cols}


# ==========================================
# RCAEval entry point
# ==========================================

def gabanzhaf(data, inject_time=None, dataset=None, num_loop=None,
              sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf with metric evidence, weighted topology and trace gating.

    The same pipeline is used for small/large systems and with/without traces.
    Missing modalities receive zero reliability instead of switching to a
    separate ranking path.

    Service-level GA approximates Banzhaf power indices while final ranking
    keeps metric-local evidence so metric-level AC@k is not collapsed to a
    service-only score.
    """
    args_ns  = kwargs.get("args")
    case_dir = (os.path.dirname(args_ns.data_path)
                if args_ns and hasattr(args_ns, "data_path") else None)
    _t_inject    = inject_time if inject_time is not None else 0
    trace_scores = _trace_onset_scores(case_dir, _t_inject)
    traces_path  = os.path.join(case_dir, "traces.csv") if case_dir else None
    call_graph   = _extract_call_graph(traces_path)

    kwargs["trace_scores"] = trace_scores
    kwargs["call_graph"]   = call_graph
    kwargs["sli"] = sli
    return _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs)
