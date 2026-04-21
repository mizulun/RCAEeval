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


def _redundancy_filtering(df, max_features=150):
    """Agglomerative clustering to reduce metric dimensionality."""
    df = df.dropna(axis=1)
    if df.shape[1] <= max_features:
        return df
    corr = df.corr().abs().fillna(0)
    dist = (1 - corr).clip(0., 1.)
    labels = AgglomerativeClustering(
        n_clusters=max_features, metric="precomputed", linkage="complete"
    ).fit_predict(dist.values)
    selected = [df.columns[labels == k][0] for k in np.unique(labels)]
    return df[selected]



def _approximate_banzhaf_ga(normal_arr, anomal_arr, n_generations=15,
                            pop_size=40, ema_alpha=0.2, rng=None):
    """
    GA-Banzhaf with MMR Fitness, Guided Mutation (delayed), and Linear Penalty.
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

    # GA 參數
    p_min, p_max = 0.1, 0.8
    alpha_mmr = 0.15        # 降低冗餘懲罰，避免過度過濾同服務相關 metrics
    warmup_gens = n_generations // 3  # 前 1/3 代使用均勻取樣，避免早期偏差固化

    for gen in range(n_generations):
        # ── Guided Mutation（warmup 期後才啟用）──
        if gen < warmup_gens or ema_banzhaf.max() == 0:
            # warmup 期間用均勻 p=0.5，保持充分探索
            p_selection = np.full(num_metrics, 0.5)
        else:
            b_norm = (ema_banzhaf - ema_banzhaf.min()) / (ema_banzhaf.max() - ema_banzhaf.min() + 1e-9)
            p_selection = p_min + (p_max - p_min) * b_norm

        pop = (rng.random((pop_size, num_metrics)) < p_selection).astype(float)
        fitness = np.zeros(pop_size)

        # 線性增長的稀疏懲罰，最大值限制在 0.1
        dynamic_penalty_weight = min(0.1, 0.05 * (1.0 + gen / n_generations))

        for i in range(pop_size):
            sel = np.where(pop[i] == 1)[0]
            if len(sel) == 0:
                continue

            coverage = float(anom_scores[sel].sum()) / denom

            redundancy = 0.0
            if len(sel) > 1:
                sub_corr = corr_matrix[np.ix_(sel, sel)]
                redundancy = (np.sum(sub_corr) - len(sel)) / (len(sel) * (len(sel) - 1))

            sparsity_penalty = len(sel) * dynamic_penalty_weight / max(1, num_metrics)

            fitness[i] = coverage - (alpha_mmr * redundancy) - sparsity_penalty

        current = np.zeros(num_metrics)
        for m in range(num_metrics):
            w = pop[:, m] == 1
            wo = ~w
            if w.sum() > 0 and wo.sum() > 0:
                current[m] = max(0., float(fitness[w].mean()) - float(fitness[wo].mean()))

        ema_banzhaf = ema_alpha * current + (1 - ema_alpha) * ema_banzhaf

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
    reduced  = _redundancy_filtering(no_time, max_features=max_features)
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
        svc_banzhaf[svc] = svc_banzhaf.get(svc, 0.) + bw

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
        svc_names=svc_names,
        topology_edges=topology_edges,
    )


def _rank_from_svc_scores(all_cols, svc_scores, metric_banzhaf):
    def _col_score(col):
        svc = _parse_svc(col)
        return float(svc_scores.get(svc, 0.)) * 0.6 + float(metric_banzhaf.get(col, 0.)) * 0.4
    return sorted(all_cols, key=_col_score, reverse=True)


# ==========================================
# RCAEval entry points
# ==========================================

def gabanzhaf(data, inject_time=None, dataset=None, num_loop=None,
              sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf + GAT re-ranking."""
    core = _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs)
    svc_scores = _gat_ranking(
        core["svc_names"], core["svc_banzhaf"], core["topology_edges"]
    )
    ranks = _rank_from_svc_scores(
        core["all_cols"], svc_scores, core["metric_banzhaf"]
    )
    return {"ranks": ranks, "node_names": core["all_cols"]}


def gabanzhaf_nogat(data, inject_time=None, dataset=None, num_loop=None,
                    sli=None, anomalies=None, **kwargs):
    """GA-Banzhaf only — no GAT re-ranking (ablation baseline)."""
    core = _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs)
    ranks = _rank_from_svc_scores(
        core["all_cols"], core["svc_banzhaf"], core["metric_banzhaf"]
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
