"""Banzhaf RCA: inference-only GAT + continuous cooperative game utility."""
from __future__ import annotations
import os, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from RCAEval.io.time_series import preprocess

warnings.filterwarnings("ignore")

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def _gmm_change_point_robust(
    data: pd.DataFrame,
    warmup_frac: float = 0.30,
    fpr: float = 5.0  # 容許 5% 的假陽性
) -> int:
    num_cols = [c for c in data.columns if c not in _NON_SLI_COLUMNS and pd.api.types.is_numeric_dtype(data[c])]
    if not num_cols: return -1

    X = np.nan_to_num(data[num_cols].values, nan=0.0)
    T = len(X)
    warmup = max(10, int(T * warmup_frac))
    X_train = X[:warmup]

    # [關鍵修正] 必須標準化並降維，否則 N_samples < N_features 時 GMM 必當機
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    n_comp = min(5, len(X_train) - 1, len(num_cols))
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_scaled)

    try:
        # 由於數據量少，建議 n_components 設為 1，退化為多元高斯分佈
        gmm = GaussianMixture(n_components=1, covariance_type='diag', random_state=42)
        gmm.fit(X_train_pca)
    except Exception:
        return -1

    train_scores = gmm.score_samples(X_train_pca)
    threshold = np.percentile(train_scores, fpr)

    # 檢測後續數據
    X_test_pca = pca.transform(scaler.transform(X[warmup:]))
    test_scores = gmm.score_samples(X_test_pca)
    
    anomalies = np.where(test_scores < threshold)[0]
    if len(anomalies) > 0:
        return warmup + anomalies[0]
    return -1

# ── Basic utilities ──────────────────────────────────────────────────────────

def _to_1d_float(x):
    return np.nan_to_num(np.asarray(x, dtype=float).reshape(-1), nan=0., posinf=0., neginf=0.)

def _safe_corr(a, b, eps=1e-12):
    a, b = _to_1d_float(a), _to_1d_float(b)
    n = min(a.size, b.size)
    if n < 3 or np.std(a[:n]) < eps or np.std(b[:n]) < eps: return 0.
    c = float(np.corrcoef(a[:n], b[:n])[0, 1])
    return c if np.isfinite(c) else 0.

def _lead_lag_corr(x, y, max_lag=2):
    x, y = _to_1d_float(x), _to_1d_float(y)
    n = min(x.size, y.size)
    if n < 5: return 0.
    best = 0.
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:   c = _safe_corr(x[:n], y[:n])
        elif lag > 0:  c = _safe_corr(x[:n-lag], y[lag:n])
        else:          c = _safe_corr(x[-lag:n], y[:n+lag])
        best = max(best, abs(c))
    return float(best)

def _min_max_norm(values):
    if not values: return {}
    vs = np.nan_to_num(np.array(list(values.values()), dtype=float))
    lo, hi = float(vs.min()), float(vs.max())
    if hi - lo < 1e-12: return {k: 0. for k in values}
    return {k: float((v - lo) / (hi - lo)) for k, v in values.items()}

# ── SLI auto-selection ───────────────────────────────────────────────────────

_SLI_PREFER = ("lat50", "lat_50", "latency", "p50", "p95", "response_time")
_SLI_SECOND = ("_error", "error_rate", "fail_rate", "_rps", "_throughput")
_NON_SLI    = frozenset({"time", "timestamp", "Time"})

def _auto_select_sli(data, hint=None):
    cols = [c for c in data.columns if c not in _NON_SLI]
    if not cols: return None
    if hint and hint in data.columns: return hint
    def _std(c):
        try: return float(pd.to_numeric(data[c], errors="coerce").std())
        except: return 0.
    t1 = [c for c in cols if any(k in c.lower() for k in _SLI_PREFER)]
    if t1: return max(t1, key=_std)
    t2 = [c for c in cols if any(k in c.lower() for k in _SLI_SECOND)]
    if t2: return max(t2, key=_std)
    return max(cols, key=_std) or None

# ── Change-point detection ───────────────────────────────────────────────────

def _cusum_cp(series, warmup_frac=0.30, h=5.0, k=0.5):
    x = _to_1d_float(series); T = len(x)
    if T < 6: return -1
    w = max(3, int(T * warmup_frac))
    mu = float(np.mean(x[:w])); sigma = float(np.std(x[:w]))
    if sigma < 1e-9: sigma = max(float(np.std(x)), 1e-9)
    if sigma < 1e-9: return -1
    S = 0.
    for t in range(w, T):
        S = max(0., S + (x[t] - mu) / sigma - k)
        if S > h: return t
    return -1

def _detect_split(data, sli_col, h=5.0, k=0.5, min_frac=0.30, max_frac=0.80):
    T = len(data)
    lo, hi = max(1, int(T * min_frac)), min(T - 1, int(T * max_frac))
    def _try(series):
        cp = _cusum_cp(series, warmup_frac=min_frac, h=h, k=k)
        return cp if lo <= cp <= hi else -1
    cp = -1
    if sli_col and sli_col in data.columns:
        cp = _try(_to_1d_float(pd.to_numeric(data[sli_col], errors="coerce").fillna(0).values))
    if cp < 0:
        num_cols = [c for c in data.columns if c not in _NON_SLI and pd.api.types.is_numeric_dtype(data[c])]
        if num_cols:
            w = max(3, lo)
            comp = np.zeros(T)
            for col in num_cols:
                x = _to_1d_float(pd.to_numeric(data[col], errors="coerce").fillna(0).values)
                mu, sig = float(np.mean(x[:w])), max(float(np.std(x[:w])), float(np.std(x)), 1e-9)
                comp += np.abs((x - mu) / sig)
            comp /= max(len(num_cols), 1)
            cp = _try(comp)
    if cp < 0 or cp < lo: return lo
    return min(cp, hi)

# ── Service / metric parsing ─────────────────────────────────────────────────

_RESOURCE_SUFFIXES = frozenset(['_cpu','_mem','_diskio','_disk','_socket','_socketio','_io'])
_LATENCY_SUFFIXES  = frozenset(['_latency','_error','_workload','_rps','_throughput','_load'])

def _parse_svc_metric(col):
    s = str(col)
    if "_" not in s: return s, "unknown"
    return s.split("_", 1)

def _group_by_svc(cols):
    d = {}
    for c in cols:
        svc, _ = _parse_svc_metric(c)
        d.setdefault(svc, []).append(c)
    return d

def _is_host(svc):
    s = str(svc).strip()
    return bool(s) and s[0].isdigit()

def _attribute_host(host_series, micro_series, anom_start, thr=0.40):
    bonus = {}
    for _, h_full in host_series.items():
        h_anom = _to_1d_float(h_full)[anom_start:]
        h_peak = float(h_anom.max()) if h_anom.size > 0 else 0.
        if h_peak < 1.: continue
        best_svc, best_c = None, 0.
        for svc, m_full in micro_series.items():
            c = abs(float(_lead_lag_corr(_to_1d_float(h_full)[anom_start:],
                                          _to_1d_float(m_full)[anom_start:])))
            if c > best_c: best_c, best_svc = c, svc
        if best_svc and best_c >= thr:
            bonus[best_svc] = max(bonus.get(best_svc, 0.), h_peak)
    return bonus

def _resource_weight(by_svc, metric_score):
    w = {}
    for svc, cols in by_svc.items():
        res = lat = 0.
        for c in cols:
            s = float(metric_score.get(c, 0.)); cl = c.lower()
            if any(cl.endswith(sf) for sf in _RESOURCE_SUFFIXES): res = max(res, s)
            elif any(cl.endswith(sf) for sf in _LATENCY_SUFFIXES): lat = max(lat, s)
        tot = res + lat
        w[svc] = 0.5 if tot < 1e-9 else res / tot
    return w

# ── Trace-based graph construction ──────────────────────────────────────────

def _read_traces_edges(case_dir):
    if not case_dir: return None
    tp = os.path.join(case_dir, "traces.csv")
    if not os.path.exists(tp): return None
    try:
        cols = list(pd.read_csv(tp, nrows=0).columns)
        cset = set(cols)
        # Direct caller/callee columns
        for a, b in [("caller_service","callee_service"),("callerService","calleeService"),
                     ("parentServiceName","serviceName"),("source","target"),("src","dst")]:
            if a in cset and b in cset:
                edges, seen = set(), 0
                for chunk in pd.read_csv(tp, usecols=[a,b], chunksize=10000):
                    for u, v in zip(chunk[a].astype(str), chunk[b].astype(str)):
                        if u.strip() not in {"nan","none",""} and v.strip() not in {"nan","none",""}:
                            edges.add((u.strip(), v.strip()))
                    seen += len(chunk)
                    if seen >= 200000 or len(edges) >= 500: break
                return list(edges) or None
        # Span-join schema
        schema = None
        for tid, sid, pid, svc in [
            ("traceID","spanID","parentSpanID","serviceName"),
            ("trace_id","span_id","parent_span_id","service"),
            ("traceId","spanId","parentSpanId","serviceName"),
            ("trace_id","span_id","parent_id","serviceName"),
        ]:
            if {tid, sid, pid, svc}.issubset(cset):
                schema = (tid, sid, pid, svc); break
        if not schema: return None
        tid, sid, pid, svc = schema
        span_svc = {}
        for chunk in pd.read_csv(tp, usecols=[tid,sid,svc], dtype=str, chunksize=10000):
            for t, s, sv in zip(chunk[tid], chunk[sid], chunk[svc]):
                if str(sv).strip().lower() not in {"nan","none",""}:
                    span_svc[(str(t).strip(), str(s).strip())] = str(sv).strip()
            if len(span_svc) >= 100000: break
        edges, seen = set(), 0
        for chunk in pd.read_csv(tp, usecols=[tid,pid,svc], dtype=str, chunksize=10000):
            for t, p, sv in zip(chunk[tid], chunk[pid], chunk[svc]):
                t, p, sv = str(t).strip(), str(p).strip(), str(sv).strip()
                if sv.lower() in {"nan","none",""} or p.lower() in {"nan","none",""}: continue
                ps = span_svc.get((t, p), "")
                if ps and ps != sv: edges.add((ps, sv))
                if len(edges) >= 500: return list(edges)
            seen += len(chunk)
            if seen >= 200000: break
        return list(edges) or None
    except Exception:
        return None

# ── Lag-correlation graph fallback ───────────────────────────────────────────

def _lag_corr_edges(svc_series, corr_thr=0.35, max_in=3, max_out=3, margin=0.05):
    svcs = sorted(svc_series.keys())
    if len(svcs) < 2: return []
    pairs = []
    for si in svcs:
        xi = _to_1d_float(svc_series[si])
        if xi.size < 3: continue
        for sj in svcs:
            if si == sj: continue
            yj = _to_1d_float(svc_series[sj])
            n = min(xi.size, yj.size)
            if n < 3: continue
            cxy = _safe_corr(xi[:n-1], yj[1:n])
            cyx = _safe_corr(yj[:n-1], xi[1:n])
            if cxy >= corr_thr and cxy > cyx + margin:
                pairs.append((cxy - cyx, si, sj))
    pairs.sort(reverse=True, key=lambda x: x[0])
    in_deg = {s: 0 for s in svcs}; out_deg = {s: 0 for s in svcs}
    edges = []
    for _, u, v in pairs:
        if out_deg[u] >= max_out or in_deg[v] >= max_in: continue
        edges.append((u, v)); out_deg[u] += 1; in_deg[v] += 1
    return edges

def _build_nbr(edges):
    out_n, in_n = {}, {}
    for u, v in edges:
        out_n.setdefault(u, []).append(v)
        in_n.setdefault(v, []).append(u)
    return out_n, in_n

# ── K-hop subgraph extraction ────────────────────────────────────────────────

def _khop_subgraph(seeds, out_n, in_n, evidence, k=2, nmax=30):
    chosen = list(dict.fromkeys(seeds))
    visited = set(chosen); frontier = list(chosen)
    for _ in range(k):
        if len(chosen) >= nmax: break
        cands = {}
        for node in frontier:
            for nb in out_n.get(node, []) + in_n.get(node, []):
                if nb not in visited:
                    cands[nb] = max(cands.get(nb, 0.), float(evidence.get(nb, 0.)))
        if not cands: break
        next_f = []
        for nb in sorted(cands, key=cands.__getitem__, reverse=True):
            if len(chosen) >= nmax: break
            chosen.append(nb); visited.add(nb); next_f.append(nb)
        frontier = next_f
    return chosen

# ── Node features ────────────────────────────────────────────────────────────

_FA, _FR, _FO, _FS, _FP, _FI, _FE, _FD = range(8)  # anom, res, onset, sli, prop, in-deg, out-deg, dim

def _node_feats(services, w_anom, w_res, w_onset, w_sli, w_prop, out_n, in_n):
    all_s = set(services) | set(out_n) | set(in_n)
    mi = max(1, max(len(in_n.get(s, [])) for s in all_s))
    mo = max(1, max(len(out_n.get(s, [])) for s in all_s))
    feats = {}
    for s in services:
        f = np.zeros(7, dtype=float)
        f[_FA] = float(w_anom.get(s, 0.)); f[_FR] = float(w_res.get(s, 0.5))
        f[_FO] = float(w_onset.get(s, 0.)); f[_FS] = float(w_sli.get(s, 0.))
        f[_FP] = float(w_prop.get(s, 0.))
        f[_FI] = len(in_n.get(s, [])) / mi; f[_FE] = len(out_n.get(s, [])) / mo
        feats[s] = f
    return feats

# ── Inference-only GAT ───────────────────────────────────────────────────────

def _gat(node_feats, edges, num_hops=2):
    if not node_feats: return {}, {}
    svcs = sorted(node_feats.keys())
    idx = {s: i for i, s in enumerate(svcs)}
    H = np.array([node_feats[s] for s in svcs], dtype=float)
    raw = {}
    for src, dst in edges:
        ii, jj = idx.get(src, -1), idx.get(dst, -1)
        if ii < 0 or jj < 0: continue
        fd = H[jj]; fs = H[ii]
        e = (0.30*fd[_FR] + 0.25*fd[_FO] + 0.20*fd[_FA]
             + 0.15*max(0., fd[_FO] - fs[_FO]) + 0.10*fd[_FS])
        raw[(ii, jj)] = float(e)
    # softmax per source
    out_nbrs = defaultdict(list)
    for ii, jj in raw: out_nbrs[ii].append(jj)
    attn = {}
    for ii, nbrs in out_nbrs.items():
        logits = np.array([raw[(ii, jj)] for jj in nbrs]); logits -= logits.max()
        exp = np.exp(logits); tot = exp.sum() + 1e-12
        for jj, a in zip(nbrs, exp / tot): attn[(ii, jj)] = float(a)
    # message passing
    n = len(svcs)
    for _ in range(num_hops):
        msg = np.zeros_like(H); wsum = np.zeros(n)
        for (ii, jj), a in attn.items(): msg[jj] += a * H[ii]; wsum[jj] += a
        H_new = H.copy()
        for v in range(n):
            if wsum[v] > 1e-12: H_new[v] = 0.6 * H[v] + 0.4 * (msg[v] / wsum[v])
        H = np.clip(H_new, -5., 5.)
    emb = {s: H[idx[s]] for s in svcs}
    ew = {(s, d): attn.get((idx.get(s,-1), idx.get(d,-1)), 0.)
          for s, d in edges if idx.get(s,-1) >= 0 and idx.get(d,-1) >= 0}
    return emb, ew

# ── GAT influence matrix ─────────────────────────────────────────────────────

def _gat_influence(nodes, gat_ew, max_hops=3, decay=0.85):
    n = len(nodes)
    if n == 0: return np.zeros((0, 0))
    idx = {s: i for i, s in enumerate(nodes)}
    A = np.zeros((n, n))
    for (caller, callee), alpha in gat_ew.items():
        ii, jj = idx.get(caller, -1), idx.get(callee, -1)
        if ii >= 0 and jj >= 0 and alpha > 0:
            A[jj, ii] = max(A[jj, ii], float(alpha))
    inf = np.eye(n); cur = A.copy()
    for hop in range(1, max_hops + 1):
        inf = np.maximum(inf, (decay ** hop) * cur)
        cur = np.max(A[:, :, np.newaxis] * cur[np.newaxis, :, :], axis=1)
    return np.clip(inf, 0., 1.)

# ── Continuous utility u(S) ──────────────────────────────────────────────────

def _utility(S_mask, n, w_anom, w_sli, influence, edge_mat, temp_mat,
             cx=0.08, lc=0.45, la=0.25, lt=0.15, lp=0.15, denom=1.,
             sli_reach=None):
    if S_mask == 0: return 0.
    members = np.array([i for i in range(n) if (S_mask >> i) & 1], dtype=np.intp)
    max_inf = influence[np.ix_(members, np.arange(n))].max(axis=0)
    cov = min(1., float(np.dot(max_inf, w_anom)) / denom)
    src = sli_reach if sli_reach is not None else w_sli
    align = float(src[members].max())
    S_vec = np.zeros(n); S_vec[members] = 1.
    outer = np.outer(S_vec, S_vec)
    e_in_S = float(np.sum(outer * edge_mat))
    temp = float(np.sum(outer * temp_mat)) / max(1., e_in_S)
    return float(lc * cov + la * align + lt * temp - lp * cx * len(members) / max(1, n))

# ── Vectorised Banzhaf estimator ─────────────────────────────────────────────

def _mc_banzhaf(n, influence, w_anom, w_sli, edge_mat, temp_mat, mc_T, n_blocks,
                denom, rng, cx=0.08, lc=0.45, la=0.25, lt=0.15, lp=0.15, sli_reach=None):
    K = n; bs = max(10, mc_T // n_blocks); T = n_blocks * bs
    masks = (rng.random((T, K)) < 0.5).astype(float)
    _NEG = -1e9
    # Coverage
    inf_mm = np.where(masks[:, :, None] > 0.5, influence[None], _NEG)
    pmax = np.maximum.accumulate(inf_mm, axis=1)
    smax = np.maximum.accumulate(inf_mm[:, ::-1, :], axis=1)[:, ::-1, :]
    left  = np.concatenate([np.full((T, 1, K), _NEG), pmax[:, :-1, :]], axis=1)
    right = np.concatenate([smax[:, 1:, :], np.full((T, 1, K), _NEG)], axis=1)
    d_cov = np.clip(np.einsum('tiv,v->ti',
        np.maximum(0., influence[None] - np.maximum(left, right)), w_anom) / denom, 0., 1.)
    # Alignment
    src = sli_reach if sli_reach is not None else w_sli
    ca = masks * src[None]
    pa = np.maximum.accumulate(ca, axis=1)
    sa = np.maximum.accumulate(ca[:, ::-1], axis=1)[:, ::-1]
    la_ = np.concatenate([np.full((T, 1), _NEG), pa[:, :-1]], axis=1)
    ra_ = np.concatenate([sa[:, 1:], np.full((T, 1), _NEG)], axis=1)
    d_aln = np.maximum(0., src[None] - np.maximum(np.maximum(la_, ra_), 0.))
    # Temporal
    De = edge_mat + edge_mat.T; Dc = temp_mat + temp_mat.T
    ef = np.einsum('ti,ij,tj->t', masks, edge_mat, masks)
    cf = np.einsum('ti,ij,tj->t', masks, temp_mat, masks)
    MDe = masks @ De; MDc = masks @ Dc
    cre = masks * MDe; crc = masks * MDc
    e_si = np.clip(ef[:, None] - cre, 0., None); c_si = np.clip(cf[:, None] - crc, 0., None)
    e_si_p = e_si + MDe; c_si_p = c_si + MDc
    d_tmp = c_si_p / np.maximum(e_si_p, 1.) - c_si / np.maximum(e_si, 1.)
    d_u = lc * d_cov + la * d_aln + lt * d_tmp - lp * cx / max(1, K)
    return d_u.reshape(n_blocks, bs, K).mean(axis=1)

# ── Downstream node pruning ──────────────────────────────────────────────────

def _prune(services, local_in_n, w_res, w_onset, w_anom, w_comb, seeds,
           rt=0.25, ot=0.20, at=0.10, min_keep=5):
    seed_set = set(seeds); kept = []; pruned = []
    for v in services:
        if (v not in seed_set
                and float(w_res.get(v, 0.5)) < rt
                and float(w_onset.get(v, 0.)) < ot
                and float(w_anom.get(v, 0.)) < at
                and bool(local_in_n.get(v, []))):
            pruned.append(v)
        else:
            kept.append(v)
    while len(kept) < min_keep and pruned:
        best = max(pruned, key=lambda x: float(w_comb.get(x, 0.)))
        pruned.remove(best); kept.append(best)
    return kept, pruned

# ── Per-service onset detection ──────────────────────────────────────────────

def _per_svc_onset(svc_series, anom_start, h=5., k=0.5):
    res = {}
    for svc, series in svc_series.items():
        x = _to_1d_float(series)
        wf = max(0.10, min(0.90, anom_start / max(len(x), 1)))
        cp = _cusum_cp(x, warmup_frac=wf, h=h, k=k)
        res[svc] = int(cp) if cp >= 0 else anom_start
    return res

# ── Expand-parameter calibration ─────────────────────────────────────────────

def _calibrate_expand(svc_series, anom_start, sli_col, col_idx, z_full, onset_map, max_anom):
    # lam_cov: anomaly SNR
    nz = [v for s in svc_series.values() for v in _to_1d_float(s)[:anom_start].tolist()]
    az = [v for s in svc_series.values() for v in _to_1d_float(s)[anom_start:].tolist()]
    sd_n = max(float(np.std(nz)), 1e-9) if nz else 1e-9
    cov_snr = max(float(np.mean(np.abs(az))) / sd_n, 0.1) if az else 1.0
    # lam_aln: SLI disruption
    aln_snr = 0.3
    if sli_col:
        j = col_idx.get(str(sli_col), -1)
        if j >= 0:
            sn = z_full[:anom_start, j]; sa = z_full[anom_start:, j]
            aln_snr = max(float(np.std(sa)) / max(float(np.std(sn)), 1e-9), 0.1)
    # lam_tmp: onset spread
    ov = [float(onset_map.get(s, max_anom)) / max(1, max_anom)
          for s in svc_series if onset_map.get(s, max_anom) < max_anom]
    tmp_snr = max(float(np.std(ov)), 0.1) if len(ov) >= 2 else 0.3
    raw = np.clip([cov_snr, aln_snr, tmp_snr], 0.1, 3.0)
    raw = raw / raw.sum()
    raw = np.maximum(raw, 0.15); raw = raw / raw.sum()
    return 0.01, float(raw[0]*0.90), float(raw[1]*0.90), float(raw[2]*0.90), 0.10

# ── Main method ──────────────────────────────────────────────────────────────

def local_banzhaf(data, inject_time=None, dataset=None, num_loop=None,
                  sli=None, anomalies=None, **kw):

    # Hyper-parameters
    seed_m  = int(kw.get("seed_m", 7));   nmax  = int(kw.get("nmax", 30))
    topk    = int(kw.get("topk", 15));    mc_T  = int(kw.get("mc_samples", 2000))
    alpha   = float(kw.get("alpha", 0.30)); beta  = float(kw.get("beta",  0.15))
    gamma   = float(kw.get("gamma", 0.10)); delta = float(kw.get("delta", 0.25))
    eps_r   = float(kw.get("epsilon_r", 0.20)); eta = float(kw.get("eta", 0.65))
    max_lag = int(kw.get("max_lag", 2));   gat_hops = int(kw.get("gat_hops", 2))
    inf_hops = int(kw.get("inf_hops", 3)); hop_decay = float(kw.get("hop_decay", 0.85))
    cx_lam  = float(kw.get("complexity_lambda", 0.08))
    prune_ok = str(kw.get("prune_enable", "true")).lower() != "false"
    prune_mk = int(kw.get("prune_min_keep", 5))
    sli_mode = str(kw.get("sli_mode", "provided"))
    subk     = int(kw.get("subgraph_k", 2))
    use_gat  = str(kw.get("use_gat", "true")).lower() != "false"
    rng = np.random.default_rng(int(os.getenv("LOCAL_BANZHAF_SEED", 7)))

    # Resolve SLI
    if sli_mode == "auto":
        effective_sli = _auto_select_sli(data)
    elif sli_mode == "auto_first":
        effective_sli = str(sli) if sli and str(sli) in data.columns else _auto_select_sli(data, sli)
    else:
        effective_sli = sli

    # [1] Window split
    inject_time = None
    anomalies = None
    # ==========================================================

    split_source = "unknown"
    try:
        if anomalies is not None:
            sr = int(anomalies[0]); normal_raw = data.head(sr); anomal_raw = data.tail(len(data)-sr)
            split_source = "anomalies"
        elif inject_time is not None and "time" in data.columns:
            normal_raw = data[data["time"] < inject_time]
            anomal_raw = data[data["time"] >= inject_time]
            split_source = "inject_time"
        else:
            # ==========================================================
            # [關鍵修正 2] 加上 method="gmm"，迫使底層使用 GMM 來找異常點
            sr = _detect_split(data, effective_sli, method="gmm") 
            # ==========================================================
            normal_raw = data.head(sr); anomal_raw = data.tail(len(data)-sr)
            split_source = "gmm"  # 將這裡的來源標籤改為 gmm，方便你後續觀察輸出
        if len(normal_raw) == 0 or len(anomal_raw) == 0: raise ValueError("empty window")
    except Exception:
        return _fallback(data, inject_time, dataset, anomalies, **kw)
    sli = effective_sli

    # [2] Preprocess jointly (preserves fault-signal cols constant in normal)
    try:
        dk = kw.get("dk_select_useful", False)
        n_rows = len(normal_raw)
        full_proc = preprocess(pd.concat([normal_raw, anomal_raw], ignore_index=True), dataset, dk)
        normal_df = full_proc.iloc[:n_rows].reset_index(drop=True)
        anomal_df = full_proc.iloc[n_rows:].reset_index(drop=True)
        shared = [c for c in normal_df.columns if c in anomal_df.columns]
        normal_df = normal_df[shared]; anomal_df = anomal_df[shared]
        if normal_df.shape[1] == 0: raise ValueError("no cols")
        ml = min(len(normal_df), len(anomal_df))
        if ml >= 3:
            normal_df = normal_df.tail(ml); anomal_df = anomal_df.head(ml)
        full_df = pd.concat([normal_df, anomal_df], ignore_index=True)
    except Exception:
        return _fallback(data, inject_time, dataset, anomalies, **kw)

    # [3] Anomaly scoring
    all_cols = list(normal_df.columns)
    by_svc_all = _group_by_svc(all_cols)
    nn = np.nan_to_num(normal_df.values.astype(float), nan=0., posinf=0., neginf=0.)
    an = np.nan_to_num(anomal_df.values.astype(float), nan=0., posinf=0., neginf=0.)
    fn = np.nan_to_num(full_df.values.astype(float),   nan=0., posinf=0., neginf=0.)
    med = np.median(nn, axis=0)
    iqr = np.percentile(nn, 75, axis=0) - np.percentile(nn, 25, axis=0)
    sc  = np.where(iqr > 0, iqr, 1.)
    z_an_all = np.clip(np.abs((an - med) / sc), 0., 100.)
    z_fn_all = np.nan_to_num(np.clip(np.abs((fn - med) / sc), 0., 100.), nan=0.)
    ci_all = {col: j for j, col in enumerate(all_cols)}

    host_svcs  = [s for s in by_svc_all if _is_host(s)]
    metric_cols = [c for c in all_cols if not _is_host(_parse_svc_metric(c)[0])]
    kept_pos = np.array([ci_all[c] for c in metric_cols], dtype=np.intp)
    z_an = z_an_all[:, kept_pos]; z_fn = z_fn_all[:, kept_pos]
    col_idx = {col: j for j, col in enumerate(metric_cols)}
    by_svc  = {s: cols for s, cols in by_svc_all.items() if not _is_host(s)}

    metric_score = {col: float(z_an[:, j].max()) for j, col in enumerate(metric_cols)}
    svc_anom = {}; svc_rep = {}
    for svc, cols in by_svc.items():
        best_c, best_s = None, -1.
        for c in cols:
            s = float(metric_score.get(c, 0.))
            if s > best_s: best_s, best_c = s, c
        svc_anom[svc] = max(0., best_s)
        if best_c: svc_rep[svc] = best_c

    anom_start = len(normal_df)
    svc_series = {}
    for svc, cols in by_svc.items():
        idxs = [col_idx[c] for c in cols if c in col_idx]
        if idxs: svc_series[svc] = z_fn[:, idxs].max(axis=1)

    # Host metric attribution
    if host_svcs:
        h_ser = {}
        for hs in host_svcs:
            hi = [ci_all[c] for c in by_svc_all[hs] if c in ci_all]
            if hi: h_ser[hs] = z_fn_all[:, hi].max(axis=1)
        for sv, bonus in _attribute_host(h_ser, svc_series, anom_start).items():
            if sv in svc_anom: svc_anom[sv] = max(svc_anom[sv], bonus)

    # Mean-shift signals
    svc_shift = {}; svc_var = {}
    for sv, ser in svc_series.items():
        x = _to_1d_float(ser)
        if len(x) < anom_start + 1: svc_shift[sv] = 0.; svc_var[sv] = 1.; continue
        ns, as_ = x[:anom_start], x[anom_start:]
        svc_shift[sv] = max(0., float(as_.mean()) - float(ns.mean()))
        svc_var[sv]   = float(np.std(as_)) / max(float(np.std(ns)), 0.1)

    # Auto-calibrate onset + corr thresholds
    all_az = [v for s in svc_series.values() for v in _to_1d_float(s)[anom_start:] if v > 0]
    onset_thr = float(max(0.5, np.percentile(all_az, 75))) if all_az else 2.
    svcs_list = sorted(svc_series.keys())
    if len(svcs_list) >= 2:
        corrs = []
        sample = [(i, j) for i in range(len(svcs_list)) for j in range(len(svcs_list)) if i != j]
        if len(sample) > 200: sample = [sample[k] for k in rng.choice(len(sample), 200, replace=False)]
        for ii, jj in sample:
            xi = _to_1d_float(svc_series[svcs_list[ii]])
            xj = _to_1d_float(svc_series[svcs_list[jj]])
            n_ = min(xi.size, xj.size)
            if n_ >= 3: corrs.append(_safe_corr(xi[:n_-1], xj[1:n_]))
        corr_thr = float(np.clip(np.percentile(corrs, 80), 0.20, 0.80)) if corrs else 0.35
    else:
        corr_thr = 0.35

    # Pre-expansion onset + sli align
    max_anom = max(1, len(anomal_df))
    onset_pre = {}
    for sv, x in svc_series.items():
        xa = _to_1d_float(x)[anom_start:]
        idxs = np.where(xa >= onset_thr)[0]
        onset_pre[sv] = int(idxs[0]) if idxs.size else max_anom
    sli_align_pre = {}
    if sli:
        sj = col_idx.get(str(sli), -1)
        if sj >= 0:
            sli_an = z_fn[anom_start:, sj]
            for sv in svc_series:
                sli_align_pre[sv] = float(_lead_lag_corr(_to_1d_float(svc_series[sv])[anom_start:], sli_an, max_lag))
    per_onset = _per_svc_onset(svc_series, anom_start)
    first_rep = min(per_onset, key=lambda s: per_onset[s]) if per_onset else None
    seed_pre  = [s for s, _ in sorted(svc_anom.items(), key=lambda x: x[1], reverse=True)[:seed_m] if s]
    _, lam_cov, lam_aln, lam_tmp, _ = _calibrate_expand(
        svc_series, anom_start, sli, col_idx, z_fn, onset_pre, max_anom)

    # [4] Build call graph
    args_obj = kw.get("args", None)
    case_dir = os.path.dirname(str(args_obj.data_path)) if args_obj and getattr(args_obj, "data_path", None) else None
    trace_edges = _read_traces_edges(case_dir)
    all_edges = trace_edges if trace_edges is not None else _lag_corr_edges(
        svc_series, corr_thr,
        max_in=int(kw.get("max_in", 3)), max_out=int(kw.get("max_out", 3)))
    out_n, in_n = _build_nbr(all_edges)

    # [5] Seed services + local subgraph
    seeds = [s for s, _ in sorted(svc_anom.items(), key=lambda x: x[1], reverse=True)[:seed_m] if s]
    sh_thr = float(np.percentile(list(svc_shift.values()), 75)) if svc_shift else 0.
    seed_set = set(seeds)
    for sv, v in sorted(svc_shift.items(), key=lambda x: x[1], reverse=True):
        if v >= sh_thr and v > 0. and sv not in seed_set and len(seeds) < seed_m * 2:
            seeds.append(sv); seed_set.add(sv)
    if first_rep and first_rep not in seed_set: seeds.insert(0, first_rep)
    local_svcs = _khop_subgraph(seeds, out_n, in_n, svc_anom, k=subk, nmax=nmax)
    if not local_svcs: return _fallback(data, inject_time, dataset, anomalies, **kw)
    local_set = set(local_svcs)
    loc_edges = [(u, v) for u, v in all_edges if u in local_set and v in local_set]
    loc_out, loc_in = _build_nbr(loc_edges)

    # [6] Weight signals
    w_loc_raw = {s: float(svc_anom.get(s, 0.)) for s in local_svcs}
    w_loc = _min_max_norm(w_loc_raw)
    w_sli_raw = {s: 0. for s in local_svcs}
    if sli:
        sj = col_idx.get(str(sli), -1)
        if sj >= 0:
            sli_an = z_fn[anom_start:, sj]
            for sv in local_svcs:
                if sv in svc_series:
                    w_sli_raw[sv] = float(_lead_lag_corr(_to_1d_float(svc_series[sv])[anom_start:], sli_an, max_lag))
    w_sli = _min_max_norm(w_sli_raw)
    onset_idx = {}
    for sv in local_svcs:
        x = svc_series.get(sv)
        if x is None: onset_idx[sv] = max_anom; continue
        xa = _to_1d_float(x)[anom_start:]
        ii = np.where(xa >= onset_thr)[0]
        onset_idx[sv] = int(ii[0]) if ii.size else max_anom
    w_prop_raw = {}
    for sv in local_svcs:
        outs = loc_out.get(sv, [])
        if not outs: w_prop_raw[sv] = 0.; continue
        w_prop_raw[sv] = sum(1 for u in outs if onset_idx.get(sv, max_anom) <= onset_idx.get(u, max_anom)) / len(outs)
    w_prop = _min_max_norm(w_prop_raw)
    onset_sc_raw = {s: (0. if onset_idx.get(s, max_anom) >= max_anom
                        else 1. - float(onset_idx[s]) / float(max_anom)) for s in local_svcs}
    onset_sc = _min_max_norm(onset_sc_raw)
    w_res = {s: float(_resource_weight(by_svc, metric_score).get(s, 0.5)) for s in local_svcs}
    w_comb = {s: (alpha*float(w_loc.get(s,0.)) + beta*float(w_sli.get(s,0.))
                  + gamma*float(w_prop.get(s,0.)) + delta*float(onset_sc.get(s,0.))
                  + eps_r*float(w_res.get(s,0.5))) for s in local_svcs}

    # SLI service exclusion (only if NOT a primary anomaly seed)
    _sli_svc = None
    if sli: _sli_svc, _ = _parse_svc_metric(str(sli))
    _seed_pre_set = set(seed_pre)
    _sli_excl = _sli_svc is not None and _sli_svc not in _seed_pre_set
    no_sli = [s for s in local_svcs if not _sli_excl or s != _sli_svc]
    if _sli_excl and _sli_svc and _sli_svc in w_comb: w_comb[_sli_svc] = 0.

    if prune_ok:
        rt = float(np.percentile([float(w_res.get(s,0.5)) for s in no_sli], 25)) if no_sli else 0.25
        ot = float(np.percentile([float(onset_sc.get(s,0.)) for s in no_sli], 25)) if no_sli else 0.25
        at = float(np.percentile([float(w_loc.get(s,0.)) for s in no_sli], 25)) if no_sli else 0.25
        pool, pruned_svcs = _prune(no_sli, loc_in, w_res, onset_sc, w_loc, w_comb,
                                   [s for s in seeds if not _sli_excl or s != _sli_svc],
                                   rt, ot, at, prune_mk)
    else:
        pool, pruned_svcs = no_sli[:], []

    # [7] GAT
    nf = _node_feats(local_svcs, w_loc, w_res, onset_sc, w_sli, w_prop, loc_out, loc_in)
    if use_gat:
        _, gat_ew = _gat(nf, loc_edges, num_hops=gat_hops)
    else:
        gat_ew = {}
        for src in set(u for u, _ in loc_edges):
            oe = [(u, v) for u, v in loc_edges if u == src]
            for e in oe: gat_ew[e] = 1. / max(1, len(oe))

    # [8] Candidates + influence matrix
    cands = [s for s, _ in sorted(w_comb.items(), key=lambda x: x[1], reverse=True)
             if s in set(pool)][:topk]
    if not cands: cands = pool[:] or local_svcs[:]
    nodes = list(dict.fromkeys(cands)); K = len(nodes); nidx = {n: i for i, n in enumerate(nodes)}
    cand_ew = {(u, v): w for (u, v), w in gat_ew.items() if u in nidx and v in nidx}
    infl = _gat_influence(nodes, cand_ew, max_hops=inf_hops, decay=hop_decay)
    w_an_arr  = np.array([float(w_loc_raw.get(n, 0.)) for n in nodes])
    w_sl_arr  = np.array([float(w_sli_raw.get(n, 0.)) for n in nodes])
    on_arr    = np.array([float(onset_sc.get(n, 0.)) for n in nodes])
    denom_cov = float(np.sum(w_an_arr)) + 1e-12
    edge_mat  = np.zeros((K, K)); temp_mat = np.zeros((K, K))
    for u_s, v_s in loc_edges:
        ii, jj = nidx.get(u_s, -1), nidx.get(v_s, -1)
        if ii >= 0 and jj >= 0:
            edge_mat[ii, jj] = 1.
            if float(on_arr[jj]) >= float(on_arr[ii]): temp_mat[ii, jj] = 1.

    # SLI reach
    sli_reach = None
    if sli:
        sv_, _ = _parse_svc_metric(str(sli))
        if sv_ in local_set:
            inf_loc = _gat_influence(local_svcs, gat_ew, max_hops=inf_hops, decay=hop_decay)
            li = {s: i for i, s in enumerate(local_svcs)}
            sj_ = li.get(sv_, -1)
            if sj_ >= 0:
                sli_reach = np.array([inf_loc[li[c], sj_] if c in li else float(w_sli_raw.get(c, 0.))
                                      for c in nodes], dtype=float)

    # [9] Banzhaf
    def u(mask): return _utility(mask, K, w_an_arr, w_sl_arr, infl, edge_mat, temp_mat,
                                  cx_lam, 0.45, 0.25, 0.15, 0.15, denom_cov, sli_reach)
    if K == 1:
        bz = {nodes[0]: 1.}; bz_ci = {nodes[0]: (0., 1.)}; bz_std = {nodes[0]: 0.}
        stab = {"kendall_tau": 1., "top_k_overlap": 1.}
    else:
        nb = max(4, min(10, mc_T // 200))
        bp = _mc_banzhaf(K, infl, w_an_arr, w_sl_arr, edge_mat, temp_mat,
                         mc_T, nb, denom_cov, rng, cx_lam, 0.45, 0.25, 0.15, 0.15, sli_reach)
        piv = bp.mean(axis=0)
        lo_, hi_ = float(piv.min()), float(piv.max())
        sc_ = max(1e-12, hi_ - lo_)
        pn = (piv - lo_) / sc_ if hi_ - lo_ > 1e-12 else np.zeros_like(piv)
        _T95 = {1:12.71,2:4.30,3:3.18,4:2.78,5:2.57,6:2.45,7:2.36,8:2.31,9:2.26}
        tf = _T95.get(nb - 1, 2.0)
        se = (bp.std(axis=0, ddof=1) / sc_) / max(1., float(nb)**0.5)
        bz     = {nodes[i]: float(pn[i]) for i in range(K)}
        bz_ci  = {nodes[i]: (float(max(0., pn[i]-tf*se[i])), float(min(1., pn[i]+tf*se[i]))) for i in range(K)}
        bz_std = {nodes[i]: float(se[i]) for i in range(K)}
        half = max(1, nb//2)
        ra, rb = bp[:half].mean(axis=0), bp[half:].mean(axis=0)
        conc = disc = 0
        for i in range(K):
            for j in range(i+1, K):
                d = (float(ra[i])-float(ra[j])) * (float(rb[i])-float(rb[j]))
                if d > 1e-15: conc += 1
                elif d < -1e-15: disc += 1
        kt = float(conc-disc) / (conc+disc) if conc+disc > 0 else 1.
        tk = min(K, 5)
        top_a = sorted(range(K), key=lambda i: float(ra[i]), reverse=True)[:tk]
        top_b = sorted(range(K), key=lambda i: float(rb[i]), reverse=True)[:tk]
        stab = {"kendall_tau": kt, "top_k_overlap": float(len(set(top_a)&set(top_b))/max(1,tk))}

    # [10] Final score + rank
    w_cn = _min_max_norm({n: float(w_comb.get(n, 0.)) for n in nodes})
    fscore = {n: eta*float(bz.get(n,0.)) + (1.-eta)*float(w_cn.get(n,0.)) for n in nodes}
    service_rank = sorted(fscore, key=lambda x: (fscore[x], float(bz_ci.get(x,(0.,0.))[0]), float(w_cn.get(x,0.))), reverse=True)

    # [11] Map to metric strings
    ranks = []; seen = set()
    for sv in service_rank:
        rep = svc_rep.get(sv)
        if rep and rep not in seen: ranks.append(rep); seen.add(rep)
    for c, _ in sorted(metric_score.items(), key=lambda x: x[1], reverse=True):
        if c not in seen: ranks.append(c); seen.add(c)

    # [12] Best coalition
    best_coal = []; best_u = -1.
    for i in range(K):
        s = u(1 << i)
        if s > best_u: best_u = s; best_coal = [nodes[i]]
    for i in range(K):
        for j in range(i+1, K):
            s = u((1<<i)|(1<<j))
            if s > best_u: best_u = s; best_coal = [nodes[i], nodes[j]]

    top_ew = sorted([(u_s,v_s,w) for (u_s,v_s),w in gat_ew.items()], key=lambda x: x[2], reverse=True)[:10]
    return {
        "node_names": metric_cols, "ranks": ranks,
        "banzhaf_values": {n: float(bz.get(n,0.)) for n in nodes},
        "banzhaf_ci": bz_ci, "banzhaf_std": bz_std, "stability": stab,
        "best_coalition": best_coal, "best_coalition_utility": float(best_u),
        "gat_edge_weights": {f"{u_s}→{v_s}": round(w,4) for u_s,v_s,w in top_ew},
        "local_services": local_svcs, "seed_services": seeds,
        "used_traces": trace_edges is not None,
        "split_source": split_source, "effective_sli": effective_sli,
        "sli_service_excluded": _sli_svc if _sli_excl else None,
        "first_reporter": first_rep,
        "per_service_onset": {s: int(v) for s, v in per_onset.items()},
        "service_mean_shift": {s: round(float(v),4) for s, v in svc_shift.items()},
        "pruned_services": pruned_svcs, "banzhaf_pool_size": len(pool),
        "calibrated": {"onset_thr": round(float(onset_thr),4), "corr_thr": round(float(corr_thr),4),
                       "expand_rho": 0.01, "lam_cov": round(lam_cov,4),
                       "lam_aln": round(lam_aln,4), "lam_tmp": round(lam_tmp,4)},
    }

# ── Fallback ─────────────────────────────────────────────────────────────────

def _fallback(data, inject_time=None, dataset=None, anomalies=None, **kw):
    try:
        from sklearn.preprocessing import RobustScaler as RS
        if anomalies is not None:
            nd, ad = data.head(anomalies[0]), data.tail(len(data)-anomalies[0])
        elif inject_time is not None and "time" in data.columns:
            nd, ad = data[data["time"] < inject_time], data[data["time"] >= inject_time]
        else:
            sr = _detect_split(data, _auto_select_sli(data))
            nd, ad = data.head(sr), data.tail(len(data)-sr)
        nd = preprocess(nd, dataset, kw.get("dk_select_useful", False))
        ad = preprocess(ad, dataset, kw.get("dk_select_useful", False))
        shared = [c for c in nd.columns if c in ad.columns]
        nd, ad = nd[shared], ad[shared]
        scores = []
        for col in nd.columns:
            a = _to_1d_float(nd[col].values); b = _to_1d_float(ad[col].values)
            if a.size < 2 or b.size == 0: scores.append((col, 0.)); continue
            sc = RS().fit(a.reshape(-1,1))
            scores.append((col, float(np.max(np.abs(np.clip(sc.transform(b.reshape(-1,1))[:,0], 0., 100.))))))
        scores.sort(key=lambda x: x[1], reverse=True)
        return {"node_names": list(nd.columns), "ranks": [c for c,_ in scores]}
    except Exception:
        try:
            d = preprocess(data, dataset, kw.get("dk_select_useful", False))
            return {"node_names": list(d.columns), "ranks": list(d.columns)}
        except Exception:
            return {"node_names": [], "ranks": []}


banzhaf = local_banzhaf
