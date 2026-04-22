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

    _nm = normal_df.drop(columns=["time"], errors="ignore").reindex(columns=no_time.columns).values.astype(float)
    _am = anomal_df.drop(columns=["time"], errors="ignore").reindex(columns=no_time.columns).values.astype(float)

    # Fast Stage-1 pre-score: mean ± std (O(n)) to rank columns quickly.
    # Used only to select the top-2×max_features candidates; the expensive
    # robust z-score (nanmedian / IQR) is then computed on that smaller set.
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

    # Robust z-score on the reduced column set
    _med = np.nanmedian(_nm2, axis=0)
    _iqr = np.nanpercentile(_nm2, 75, axis=0) - np.nanpercentile(_nm2, 25, axis=0)
    _sc  = np.where(_iqr > 0, _iqr, 1.)
    _pre_anom_reduced = np.clip(np.abs((_am2 - _med) / _sc), 0., 100.).max(axis=0)

    # Build full _pre_anom dict: robust score for top cols, fast score for rest
    if _nm.shape[1] > max_features * 2:
        _pre_anom = dict(zip(no_time.columns,
                             np.abs((_am - _nm.mean(axis=0)) /
                                    np.where(_nm.std(axis=0) > 0, _nm.std(axis=0), 1.)).max(axis=0)))
        for col, score in zip(_cols2, _pre_anom_reduced):
            _pre_anom[col] = float(score)   # overwrite with robust score
    else:
        _pre_anom = dict(zip(_cols2, _pre_anom_reduced))

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

    # ── 5. Build topology from combined data ─────────────────────────────────
    # Combined (normal + anomaly) data preserves fault-propagation edges that
    # PPR needs to trace back to the root cause.  Using only normal-period data
    # removes these edges and hurts PPR recall.
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


