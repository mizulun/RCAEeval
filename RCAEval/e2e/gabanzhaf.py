import warnings
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

from RCAEval.io.time_series import preprocess


# ==========================================
# Module 1: Banzhaf-Aware GAT
# ==========================================
class BanzhafAwareGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=8):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, 1, heads=1, concat=False)

    def forward(self, data, banzhaf_weights):
        x, edge_index = data.x, data.edge_index
        banzhaf_tensor = torch.tensor(banzhaf_weights, dtype=torch.float32).view(-1, 1)
        x = x * banzhaf_tensor
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


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

    # Stage 2: agglomerative clustering
    corr = df.corr().abs().fillna(0)
    dist = (1 - corr).clip(0., 1.)
    labels = AgglomerativeClustering(
        n_clusters=max_features, metric="precomputed", linkage="complete"
    ).fit_predict(dist.values)
    if anom_scores is not None:
        selected = [
            max(df.columns[labels == k], key=lambda c: anom_scores.get(c, 0.))
            for k in np.unique(labels)
        ]
    else:
        selected = [df.columns[labels == k][0] for k in np.unique(labels)]
    return df[selected]



def _eval_fitness(pop, anom_scores, denom, corr_matrix, num_metrics,
                  alpha_mmr, penalty_weight):
    """Evaluate fitness for all individuals in the population."""
    fitness = np.zeros(len(pop))
    for i, ind in enumerate(pop):
        sel = np.where(ind == 1)[0]
        if len(sel) == 0:
            continue
        coverage = float(anom_scores[sel].sum()) / denom
        redundancy = 0.0
        if len(sel) > 1:
            sub_corr = corr_matrix[np.ix_(sel, sel)]
            redundancy = (np.sum(sub_corr) - len(sel)) / (len(sel) * (len(sel) - 1))
        sparsity_penalty = len(sel) * penalty_weight / max(1, num_metrics)
        fitness[i] = coverage - alpha_mmr * redundancy - sparsity_penalty
    return fitness


def _tournament_select(pop, fitness, rng, k=3):
    """Tournament selection: pick best among k random individuals."""
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[idx[np.argmax(fitness[idx])]].copy()


def _approximate_banzhaf_ga(normal_arr, anomal_arr, n_generations=15,
                            pop_size=40, ema_alpha=0.2, rng=None):
    """
    GA-Banzhaf with Elitism, Tournament Selection, Crossover, and Guided Mutation.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    med = np.median(normal_arr, axis=0)
    iqr = np.percentile(normal_arr, 75, axis=0) - np.percentile(normal_arr, 25, axis=0)
    sc = np.where(iqr > 0, iqr, 1.)
    z_an = np.clip(np.abs((anomal_arr - med) / sc), 0., 100.)
    anom_scores = z_an.max(axis=0)
    denom = max(1., float(anom_scores.sum()))

    anom_df = pd.DataFrame(anomal_arr)
    corr_matrix = anom_df.corr().abs().fillna(0).values

    num_metrics = normal_arr.shape[1]
    ema_banzhaf = np.zeros(num_metrics)

    alpha_mmr   = 0.15
    elite_k     = max(2, pop_size // 10)   # 精英數量：約 10%
    p_mut_base  = 1.0 / max(1, num_metrics) # 每個 bit 的基礎突變率
    warmup_gens = n_generations // 3

    # 初始族群：均勻隨機
    pop = (rng.random((pop_size, num_metrics)) < 0.5).astype(float)

    for gen in range(n_generations):
        # 線性增長的稀疏懲罰，上限 0.1
        penalty_weight = min(0.1, 0.05 * (1.0 + gen / n_generations))

        fitness = _eval_fitness(pop, anom_scores, denom, corr_matrix,
                                num_metrics, alpha_mmr, penalty_weight)

        # ── Banzhaf 邊際貢獻估計 ──
        current = np.zeros(num_metrics)
        for m in range(num_metrics):
            w  = pop[:, m] == 1
            wo = ~w
            if w.sum() > 0 and wo.sum() > 0:
                current[m] = max(0., float(fitness[w].mean()) - float(fitness[wo].mean()))
        ema_banzhaf = ema_alpha * current + (1 - ema_alpha) * ema_banzhaf

        # ── 精英保留 ──
        elite_idx = np.argsort(fitness)[-elite_k:]
        next_pop  = list(pop[elite_idx])

        # ── Guided Mutation 參數（warmup 後啟用）──
        if gen >= warmup_gens and ema_banzhaf.max() > 0:
            b_norm = ((ema_banzhaf - ema_banzhaf.min()) /
                      (ema_banzhaf.max() - ema_banzhaf.min() + 1e-9))
            # 低 Banzhaf 的 bit：較高翻轉率；高 Banzhaf 的 bit：保留
            p_flip = p_mut_base * (1.0 + (1.0 - b_norm))
        else:
            p_flip = np.full(num_metrics, p_mut_base * 2)

        # ── Tournament Selection + Uniform Crossover + Mutation ──
        while len(next_pop) < pop_size:
            p1 = _tournament_select(pop, fitness, rng)
            p2 = _tournament_select(pop, fitness, rng)

            # Uniform crossover
            mask  = rng.random(num_metrics) < 0.5
            child = np.where(mask, p1, p2).copy()

            # Bit-flip mutation
            flip  = rng.random(num_metrics) < p_flip
            child[flip] = 1.0 - child[flip]

            # 確保至少有一個特徵被選中
            if child.sum() == 0:
                child[rng.integers(num_metrics)] = 1.0

            next_pop.append(child)

        pop = np.array(next_pop[:pop_size])

    return ema_banzhaf


def _build_topology(svc_names, svc_series, corr_thr=0.30):
    """Build directed topology from lag-1 cross-correlations between services."""
    edges = []
    svcs = list(svc_names)
    for i, s1 in enumerate(svcs):
        for j, s2 in enumerate(svcs):
            if i == j:
                continue
            x1 = svc_series.get(s1)
            x2 = svc_series.get(s2)
            if x1 is None or x2 is None:
                continue
            n = min(len(x1), len(x2)) - 1
            if n < 3:
                continue
            c = float(np.corrcoef(x1[:n], x2[1:n + 1])[0, 1])
            if c > corr_thr:
                edges.append((s1, s2))
    return edges


def _build_pyg_data(node_list, svc_banzhaf, topology_edges, device):
    """Build shared PyG Data object and node_banzhaf list."""
    G = nx.DiGraph()
    G.add_nodes_from(node_list)
    G.add_edges_from([(u, v) for u, v in topology_edges
                      if u in node_list and v in node_list])
    node_idx = {n: i for i, n in enumerate(node_list)}

    node_features, node_banzhaf = [], []
    for nd in node_list:
        bw = float(svc_banzhaf.get(nd, 0.))
        node_features.append([bw, bw ** 2, float(bw > 0), 1.])
        node_banzhaf.append(bw)

    x = torch.tensor(node_features, dtype=torch.float32).to(device)
    edges = list(G.edges())
    if edges:
        edge_index = torch.tensor(
            [[node_idx[u], node_idx[v]] for u, v in edges], dtype=torch.long
        ).t().contiguous().to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)

    return Data(x=x, edge_index=edge_index), node_banzhaf


def _gat_ranking(node_list, svc_banzhaf, topology_edges):
    """Random-init GAT re-ranking (no training)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, node_banzhaf = _build_pyg_data(node_list, svc_banzhaf, topology_edges, device)
    model = BanzhafAwareGAT(num_node_features=4).to(device)
    model.eval()
    with torch.no_grad():
        scores = model(data, node_banzhaf).cpu().numpy().flatten()
    return {node_list[i]: float(scores[i]) for i in range(len(node_list))}


def _gat_ranking_ssl(node_list, svc_banzhaf, topology_edges, n_epochs=50, lr=0.01):
    """
    Self-supervised GAT: train to reconstruct normalised Banzhaf scores
    as pseudo-labels (transductive, on the current graph), then re-rank.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, node_banzhaf = _build_pyg_data(node_list, svc_banzhaf, topology_edges, device)

    # Pseudo-labels: min-max normalised Banzhaf scores
    bz = torch.tensor(node_banzhaf, dtype=torch.float32).to(device)
    bz_min, bz_max = bz.min(), bz.max()
    labels = (bz - bz_min) / (bz_max - bz_min + 1e-9)

    model = BanzhafAwareGAT(num_node_features=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        out = model(data, node_banzhaf).squeeze()
        loss = F.mse_loss(out, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = model(data, node_banzhaf).cpu().numpy().flatten()
    return {node_list[i]: float(scores[i]) for i in range(len(node_list))}


# ==========================================
# Shared core pipeline (steps 1–5)
# ==========================================

def _svc_series_from_df(df, svc_names):
    """Aggregate per-service time series (max across metrics) from a DataFrame."""
    result = {}
    for svc in svc_names:
        cols = [c for c in df.columns if _parse_svc(c) == svc]
        if cols:
            result[svc] = df[cols].max(axis=1).values.astype(float)
    return result


def _structural_change_score(norm_svc_series, anom_svc_series):
    """Per-service structural change score.

    Measures how much each service's pairwise correlation pattern with
    other services shifted between the normal and anomaly windows.

        change_score(s) = mean_t≠s |corr_anom(s,t) − corr_norm(s,t)|

    A high score means s's relationships changed significantly —
    consistent with it being near the fault injection point.
    """
    svcs = sorted(
        s for s in norm_svc_series if s in anom_svc_series
    )
    n = len(svcs)
    if n < 2:
        return {s: 0. for s in svcs}

    def _corr_mat(series_dict):
        mat = np.zeros((n, n))
        for i, s1 in enumerate(svcs):
            for j, s2 in enumerate(svcs):
                if i == j:
                    mat[i, j] = 1.
                    continue
                x1, x2 = series_dict[s1], series_dict[s2]
                length = min(len(x1), len(x2))
                if length < 3:
                    continue
                c = np.corrcoef(x1[:length], x2[:length])[0, 1]
                mat[i, j] = float(c) if np.isfinite(c) else 0.
        return mat

    diff = np.abs(_corr_mat(anom_svc_series) - _corr_mat(norm_svc_series))
    change = {}
    for i, svc in enumerate(svcs):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        change[svc] = float(diff[i, mask].mean()) if mask.any() else 0.
    return change


def _prefilter_services(all_svc_names, svc_change, svc_anom, normal_edges,
                         top_k=None, change_w=0.5):
    """Select candidate services before GA-Banzhaf.

    Strategy:
      1. Score each service: change_w × change_norm + (1−change_w) × anom_norm
      2. Keep the top-K services by combined score
      3. Expand by adding all direct neighbors (predecessors & successors) in
         the normal-period topology — upstream root causes may be less anomalous
         but are adjacent to anomalous victims

    Returns a set of service names to keep.
    """
    n = len(all_svc_names)
    if top_k is None:
        top_k = max(5, n // 2)

    def _norm(d):
        vals = list(d.values())
        lo, hi = min(vals, default=0.), max(vals, default=1.)
        span = hi - lo if hi > lo else 1.
        return {k: (v - lo) / span for k, v in d.items()}

    nc = _norm({s: svc_change.get(s, 0.) for s in all_svc_names})
    na = _norm({s: svc_anom.get(s, 0.)   for s in all_svc_names})
    combined = {s: change_w * nc[s] + (1. - change_w) * na[s]
                for s in all_svc_names}

    ranked   = sorted(all_svc_names, key=lambda s: combined[s], reverse=True)
    selected = set(ranked[:top_k])

    # Expand: add all neighbors from the normal-period graph
    adj: dict = {s: set() for s in all_svc_names}
    for u, v in normal_edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)
    for svc in list(selected):
        selected.update(adj.get(svc, set()))

    return selected


def _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs):
    """
    Shared pipeline: window split → preprocess → dim-reduction → GA-Banzhaf → topology.

    Returns a dict with all intermediate results needed by the entry points.
    """
    n_generations = int(kwargs.get("n_generations", 15))
    pop_size      = int(kwargs.get("pop_size", 40))
    ema_alpha     = float(kwargs.get("ema_alpha", 0.2))
    max_features  = int(kwargs.get("max_features", 150))
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

    # ── 3. Dimensionality reduction ──────────────────────────────────────────
    full_df = pd.concat([normal_df, anomal_df], ignore_index=True)
    no_time = full_df.drop(columns=["time"], errors="ignore")

    # Pre-compute per-column anomaly scores for anomaly-aware cluster selection.
    # Use robust z-score on the anomaly window only.
    _nm = normal_df.drop(columns=["time"], errors="ignore").reindex(columns=no_time.columns).values.astype(float)
    _am = anomal_df.drop(columns=["time"], errors="ignore").reindex(columns=no_time.columns).values.astype(float)
    _med = np.nanmedian(_nm, axis=0)
    _iqr = np.nanpercentile(_nm, 75, axis=0) - np.nanpercentile(_nm, 25, axis=0)
    _sc  = np.where(_iqr > 0, _iqr, 1.)
    _pre_anom = dict(zip(
        no_time.columns,
        np.clip(np.abs((_am - _med) / _sc), 0., 100.).max(axis=0)
    ))

    reduced  = _redundancy_filtering(no_time, max_features=max_features, anom_scores=_pre_anom)

    # ── Force service coverage ───────────────────────────────────────────────
    # For any service not represented after filtering, add its single most
    # anomalous metric so GA can still evaluate it.
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

    # svc_banzhaf: max over per-service metric Banzhaf values
    svc_banzhaf: dict = {}
    for col, bw in metric_banzhaf.items():
        svc = _parse_svc(col)
        svc_banzhaf[svc] = max(svc_banzhaf.get(svc, 0.), bw)

    # svc_anom: max anomaly score per service from ALL original columns
    # (pre-filtering), so every service is covered even if it has no
    # post-filtering representative.
    svc_anom: dict = {}
    for col, score in _pre_anom.items():
        svc = _parse_svc(col)
        svc_anom[svc] = max(svc_anom.get(svc, 0.), score)

    # ── 5. Build topology ────────────────────────────────────────────────────
    svc_names = sorted(set(_parse_svc(c) for c in all_cols))
    svc_series = {}
    for svc in svc_names:
        idxs = [i for i, c in enumerate(red_cols) if _parse_svc(c) == svc]
        if idxs:
            svc_series[svc] = reduced.values[:, idxs].max(axis=1).astype(float)

    topology_edges = _build_topology(svc_names, svc_series)

    return dict(
        all_cols=all_cols,
        metric_banzhaf=metric_banzhaf,
        svc_banzhaf=svc_banzhaf,
        svc_anom=svc_anom,
        svc_names=svc_names,
        topology_edges=topology_edges,
    )


def _gabanzhaf_core_prefilt(data, inject_time, dataset, anomalies, kwargs):
    """
    Pre-filtering pipeline:
      Step 0  Build normal-period service topology & structural change scores
      Step 0b Pre-filter: keep top-K services + their graph neighbors
      Step 1-4 Same as _gabanzhaf_core, but GA-Banzhaf runs only on
               metrics belonging to the candidate services
      Step 5  Use the NORMAL-period topology for PPR (stable baseline)

    Returns the same dict as _gabanzhaf_core, plus 'normal_topology_edges'.
    """
    n_generations = int(kwargs.get("n_generations", 15))
    pop_size      = int(kwargs.get("pop_size", 40))
    ema_alpha     = float(kwargs.get("ema_alpha", 0.2))
    max_features  = int(kwargs.get("max_features", 150))
    top_k         = kwargs.get("prefilt_top_k", None)   # None → n//2
    change_w      = float(kwargs.get("prefilt_change_w", 0.5))
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

    all_cols  = list(normal_df.columns)
    all_svcs  = sorted(set(_parse_svc(c) for c in all_cols))

    no_time_n = normal_df.drop(columns=["time"], errors="ignore")
    no_time_a = anomal_df.drop(columns=["time"], errors="ignore")

    # ── Step 0: Pre-filter services ──────────────────────────────────────────
    # 0a. Anomaly score per service (robust z-score, all columns)
    _nm  = no_time_n.values.astype(float)
    _am  = no_time_a.values.astype(float)
    _med = np.nanmedian(_nm, axis=0)
    _iqr = np.nanpercentile(_nm, 75, axis=0) - np.nanpercentile(_nm, 25, axis=0)
    _sc  = np.where(_iqr > 0, _iqr, 1.)
    _pre_anom = dict(zip(
        no_time_n.columns,
        np.clip(np.abs((_am - _med) / _sc), 0., 100.).max(axis=0)
    ))
    svc_anom_full: dict = {}
    for col, score in _pre_anom.items():
        svc = _parse_svc(col)
        svc_anom_full[svc] = max(svc_anom_full.get(svc, 0.), score)

    # 0b. Normal-period topology (from normal data only)
    norm_svc_series = _svc_series_from_df(no_time_n, all_svcs)
    anom_svc_series = _svc_series_from_df(no_time_a, all_svcs)
    normal_edges    = _build_topology(all_svcs, norm_svc_series)

    # 0c. Structural change score
    svc_change = _structural_change_score(norm_svc_series, anom_svc_series)

    # 0d. Select candidate services
    candidate_svcs = _prefilter_services(
        all_svcs, svc_change, svc_anom_full, normal_edges,
        top_k=top_k, change_w=change_w,
    )

    # 0e. Keep only metrics belonging to candidate services
    cand_cols = [c for c in all_cols if _parse_svc(c) in candidate_svcs]
    if not cand_cols:                    # safety: keep everything if filter nukes all
        cand_cols = all_cols

    # ── 3. Dimensionality reduction (on candidate metrics only) ──────────────
    full_df  = pd.concat([
        normal_df[cand_cols].reset_index(drop=True),
        anomal_df[cand_cols].reset_index(drop=True),
    ], ignore_index=True)
    no_time  = full_df.drop(columns=["time"], errors="ignore")

    # Re-compute anomaly scores restricted to candidate columns
    _nm2 = no_time.values[:ml].astype(float)
    _am2 = no_time.values[ml:].astype(float)
    _med2 = np.nanmedian(_nm2, axis=0)
    _iqr2 = np.nanpercentile(_nm2, 75, axis=0) - np.nanpercentile(_nm2, 25, axis=0)
    _sc2  = np.where(_iqr2 > 0, _iqr2, 1.)
    _cand_anom = dict(zip(
        no_time.columns,
        np.clip(np.abs((_am2 - _med2) / _sc2), 0., 100.).max(axis=0)
    ))

    reduced  = _redundancy_filtering(no_time, max_features=max_features,
                                      anom_scores=_cand_anom)

    # Force coverage for every candidate service
    covered_svcs = set(_parse_svc(c) for c in reduced.columns)
    svc_to_cols: dict = {}
    for c in no_time.columns:
        svc_to_cols.setdefault(_parse_svc(c), []).append(c)

    extra_cols = []
    for svc, cols in svc_to_cols.items():
        if svc not in covered_svcs:
            best = max(cols, key=lambda c: _cand_anom.get(c, 0.))
            extra_cols.append(best)
    if extra_cols:
        reduced = pd.concat([reduced, no_time[extra_cols]], axis=1)

    red_cols   = list(reduced.columns)
    red_normal = reduced.values[:ml].astype(float)
    red_anomal = reduced.values[ml:].astype(float)

    # ── 4. GA-Banzhaf (focused on candidate metrics) ─────────────────────────
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

    # For non-candidate services: give them 0 Banzhaf / their original anom score
    for svc in all_svcs:
        if svc not in svc_banzhaf:
            svc_banzhaf[svc] = 0.

    return dict(
        all_cols=all_cols,
        metric_banzhaf=metric_banzhaf,
        svc_banzhaf=svc_banzhaf,
        svc_anom=svc_anom_full,          # full coverage, all services
        svc_names=all_svcs,
        topology_edges=normal_edges,     # normal-period topology for PPR
        candidate_svcs=candidate_svcs,
        svc_change=svc_change,
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


# ==========================================
# Causal-topology adjustments (Steps A/B/C)
# ==========================================

def _upstream_damping(svc_scores, topology_edges, beta=0.5):
    """方案一：Upstream Damping.

    Dampen a service's score proportionally to how much of the total
    anomaly signal sits in its direct predecessors.  If A→B and A is
    highly scored, B's score is reduced because B's anomaly is likely
    *caused* by A rather than being a root cause itself.
    """
    total = sum(svc_scores.values()) or 1.
    # build predecessor map
    predecessors: dict = {s: [] for s in svc_scores}
    for u, v in topology_edges:
        if u in svc_scores and v in predecessors:
            predecessors[v].append(u)

    damped = {}
    for svc, score in svc_scores.items():
        upstream_sum = sum(svc_scores.get(p, 0.) for p in predecessors[svc])
        ratio = upstream_sum / total          # how much upstream explains
        damped[svc] = score * (1. - beta * ratio)
    return damped


def _reverse_pagerank(svc_scores, topology_edges, alpha=0.85, max_iter=200,
                      svc_anom=None, anom_weight=0.3):
    """方案二：Personalized PageRank on the reversed causal graph.

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


def _topo_penalty(svc_scores, topology_edges, gamma=0.5):
    """方案三：Topological Penalty.

    Penalise services whose predecessors already have high scores.
    The penalty per predecessor is its score divided by its out-degree
    (i.e. how much causal "blame" it can distribute).
    """
    predecessors: dict = {s: [] for s in svc_scores}
    out_degree: dict = {s: 0 for s in svc_scores}
    for u, v in topology_edges:
        if u in svc_scores and v in predecessors:
            predecessors[v].append(u)
        if u in out_degree:
            out_degree[u] += 1

    penalized = {}
    for svc, score in svc_scores.items():
        penalty = sum(
            svc_scores.get(p, 0.) / max(1, out_degree.get(p, 1))
            for p in predecessors[svc]
        )
        penalized[svc] = score - gamma * penalty
    return penalized


# ==========================================
# RCAEval entry points
# ==========================================

def gabanzhaf(data, inject_time=None, dataset=None, num_loop=None,
              sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf + Reversed Personalized PageRank (main method).

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


def gabanzhaf_nogat(data, inject_time=None, dataset=None, num_loop=None,
                    sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf only — no GAT re-ranking (ablation baseline)."""
    core = _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs)
    ranks = _rank_from_svc_scores(
        core["all_cols"], core["svc_banzhaf"],
        core["metric_banzhaf"], svc_anom=core["svc_anom"]
    )
    return {"ranks": ranks, "node_names": core["all_cols"]}


def gabanzhaf_ssl(data, inject_time=None, dataset=None, num_loop=None,
                  sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf + self-supervised GAT (Banzhaf scores as pseudo-labels)."""
    n_epochs = int(kwargs.get("ssl_epochs", 50))
    lr       = float(kwargs.get("ssl_lr", 0.01))
    core = _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs)
    svc_scores = _gat_ranking_ssl(
        core["svc_names"], core["svc_banzhaf"], core["topology_edges"],
        n_epochs=n_epochs, lr=lr,
    )
    ranks = _rank_from_svc_scores(
        core["all_cols"], svc_scores, core["metric_banzhaf"]
    )
    return {"ranks": ranks, "node_names": core["all_cols"]}


def gabanzhaf_upstream(data, inject_time=None, dataset=None, num_loop=None,
                       sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf + Upstream Damping (方案一).

    Dampen a service's Banzhaf score when its direct predecessors in the
    causal topology already carry high scores — the service is likely a
    downstream victim rather than the root cause.
    """
    beta = float(kwargs.get("upstream_beta", 0.5))
    core = _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs)
    svc_scores = _upstream_damping(
        core["svc_banzhaf"], core["topology_edges"], beta=beta
    )
    ranks = _rank_from_svc_scores(
        core["all_cols"], svc_scores, core["metric_banzhaf"],
        svc_anom=core["svc_anom"]
    )
    return {"ranks": ranks, "node_names": core["all_cols"]}


def gabanzhaf_pagerank(data, inject_time=None, dataset=None, num_loop=None,
                       sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf + Reversed Personalized PageRank (方案二, ablation).

    Run Personalized PageRank on the reversed causal graph, seeded by
    Banzhaf scores.  Nodes that can causally explain many downstream
    anomalies receive high probability mass.
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


def gabanzhaf_topo(data, inject_time=None, dataset=None, num_loop=None,
                   sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf + Topological Penalty (方案三).

    Penalise services whose causal predecessors are already highly scored,
    redistributing "blame" toward upstream root causes.
    """
    gamma = float(kwargs.get("topo_gamma", 0.5))
    core  = _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs)
    svc_scores = _topo_penalty(
        core["svc_banzhaf"], core["topology_edges"], gamma=gamma
    )
    ranks = _rank_from_svc_scores(
        core["all_cols"], svc_scores, core["metric_banzhaf"],
        svc_anom=core["svc_anom"]
    )
    return {"ranks": ranks, "node_names": core["all_cols"]}


def gabanzhaf_prefilt(data, inject_time=None, dataset=None, num_loop=None,
                      sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf with pre-injection service pre-filtering + normal-period PPR.

    Key difference from gabanzhaf:
      1. Builds normal-period topology BEFORE GA-Banzhaf runs
      2. Computes structural change score per service (corr pattern shift)
      3. Pre-filters to top-K services + their graph neighbors
      4. GA-Banzhaf runs only on metrics of candidate services (less noise)
      5. PPR uses the stable normal-period topology as causal backbone
    """
    alpha       = float(kwargs.get("pr_alpha", 0.85))
    anom_weight = float(kwargs.get("pr_anom_weight", 0.3))
    core = _gabanzhaf_core_prefilt(data, inject_time, dataset, anomalies, kwargs)
    svc_scores = _reverse_pagerank(
        core["svc_banzhaf"], core["topology_edges"], alpha=alpha,
        svc_anom=core["svc_anom"], anom_weight=anom_weight,
    )
    ranks = _rank_from_svc_scores(
        core["all_cols"], svc_scores, core["metric_banzhaf"],
        svc_anom=core["svc_anom"]
    )
    return {"ranks": ranks, "node_names": core["all_cols"]}
