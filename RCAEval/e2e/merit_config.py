"""MERIT defaults and validated keyword configuration; no runtime environment reads."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _GabanzhafConfig:
    """MERIT configuration; the historical class name is retained for callers."""

    top_k_svcs: int = 15
    dk_select_useful: bool = False
    exact_candidate_k: int = 12
    banzhaf_mode: str = "closed_form"
    rank_mode: str = "banzhaf"
    early_window_ratio: float = 0.25
    early_window_min_len: int = 5
    use_early_full_fusion: bool = True
    # Evidence-blend weights. Hard-coded until 2026-08-31; parameterised so the
    # sensitivity sweep can cover them (they sit inside the metric evidence
    # stage, which the factorial ablation showed to be the dominant factor).
    # Each group is renormalised to sum to 1, so any non-negative value is legal.
    ev_w_mag: float = 0.55
    ev_w_persistence: float = 0.30
    ev_w_peak: float = 0.15
    early_w_mag: float = 0.70
    early_w_peak: float = 0.30
    fusion_w_early: float = 0.60
    fusion_w_full: float = 0.40
    agg_w_p95: float = 0.50
    agg_w_top3: float = 0.30
    agg_w_max: float = 0.20
    svc_w_agg: float = 0.70
    svc_w_dominance: float = 0.30
    use_metric_type_dominance: bool = True
    use_metrics: bool = True
    # Modality-level ablation gate.  When false, evidence extraction and every
    # downstream stage must treat all trace-derived inputs as absent while the
    # metric -> candidate -> game -> Banzhaf -> ranking pipeline remains intact.
    use_trace: bool = True
    # Call-graph coverage: rescue a quiet upstream root that explains several
    # loud downstream victims (the delay/loss propagation signature). Gated on
    # >=min_loud children above loud_thr so single-strong-anomaly faults
    # (cpu/mem/disk/socket) are left untouched.
    coverage_conf: float = 0.7
    coverage_loud_thr: float = 0.4
    coverage_lat_floor: float = 0.1
    coverage_min_loud: int = 2
    use_trace_topology: bool = True
    trace_topology_max_lag: int = 3
    trace_topology_min_conf: float = 0.3
    trace_topology_conf: float = 0.85
    trace_signal: str = "both"
    trace_control: str = "observed"
    trace_dropout_fraction: float = 0.0
    trace_noise_fraction: float = 0.0
    trace_edge_noise_fraction: float = 0.0
    # Causal-edge thresholds used by victim demotion (historically named
    # frontier_*; names kept to preserve the kwargs API).
    frontier_abnormal_threshold: float = 0.45
    frontier_normal_threshold: float = 0.20
    frontier_edge_threshold: float = 0.30
    frontier_uncertain_stop_factor: float = 0.50
    # Victim demotion: when an abnormal service's causal upstream is also
    # abnormal, it is a propagation victim, so shrink its self-coverage (the
    # diagonal 1.0) toward victim_floor. Roots (normal/absent upstream) keep 1.0.
    use_victim_demotion: bool = True
    protect_coverage_parents: bool = True
    victim_floor: float = 0.5
    # Residual trace extraction: build edges from the inject pre/post delta
    # instead of the full-window normalized series, so unperturbed services
    # (which co-activate but are not the propagation path) drop out.
    trace_residual: bool = False
    trace_min_energy: float = 0.1
    # Metric anomaly scorer. "robust_z" (default) keeps the magnitude/
    # persistence/peak robust z-score behaviour unchanged. "cusum" is an
    # alternative that accumulates small persistent deviations without a hard
    # rz>3 threshold, targeting slow-onset gradual-degradation faults that the
    # persistence term misses. Default-off; on/off ablation decides adoption.
    scorer: str = "robust_z"
    cusum_slack: float = 0.5

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

    @staticmethod
    def _nonnegative_int_or_default(value, default):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return int(default)
        if not np.isfinite(parsed) or parsed < 0.0:
            return int(default)
        return int(parsed)

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
            use_early_full_fusion=bool(
                kwargs.get("use_early_full_fusion", cls.use_early_full_fusion)
            ),
            use_metric_type_dominance=bool(
                kwargs.get(
                    "use_metric_type_dominance",
                    cls.use_metric_type_dominance,
                )
            ),
            ev_w_mag=float(kwargs.get("ev_w_mag", cls.ev_w_mag)),
            ev_w_persistence=float(kwargs.get("ev_w_persistence", cls.ev_w_persistence)),
            ev_w_peak=float(kwargs.get("ev_w_peak", cls.ev_w_peak)),
            early_w_mag=float(kwargs.get("early_w_mag", cls.early_w_mag)),
            early_w_peak=float(kwargs.get("early_w_peak", cls.early_w_peak)),
            fusion_w_early=float(kwargs.get("fusion_w_early", cls.fusion_w_early)),
            fusion_w_full=float(kwargs.get("fusion_w_full", cls.fusion_w_full)),
            agg_w_p95=float(kwargs.get("agg_w_p95", cls.agg_w_p95)),
            agg_w_top3=float(kwargs.get("agg_w_top3", cls.agg_w_top3)),
            agg_w_max=float(kwargs.get("agg_w_max", cls.agg_w_max)),
            svc_w_agg=float(kwargs.get("svc_w_agg", cls.svc_w_agg)),
            svc_w_dominance=float(kwargs.get("svc_w_dominance", cls.svc_w_dominance)),
            use_metrics=bool(kwargs.get("use_metrics", cls.use_metrics)),
            use_trace=bool(kwargs.get("use_trace", cls.use_trace)),
            coverage_conf=float(kwargs.get("coverage_conf", cls.coverage_conf)),
            coverage_loud_thr=float(kwargs.get("coverage_loud_thr", cls.coverage_loud_thr)),
            coverage_lat_floor=float(kwargs.get("coverage_lat_floor", cls.coverage_lat_floor)),
            coverage_min_loud=cls._positive_int_or_default(
                kwargs.get("coverage_min_loud", cls.coverage_min_loud),
                cls.coverage_min_loud,
            ),
            use_trace_topology=bool(
                kwargs.get("use_trace_topology", cls.use_trace_topology)
            ),
            trace_topology_max_lag=cls._nonnegative_int_or_default(
                kwargs.get("trace_topology_max_lag", cls.trace_topology_max_lag),
                cls.trace_topology_max_lag,
            ),
            trace_topology_min_conf=float(
                kwargs.get("trace_topology_min_conf", cls.trace_topology_min_conf)
            ),
            trace_topology_conf=float(
                kwargs.get("trace_topology_conf", cls.trace_topology_conf)
            ),
            trace_signal=str(kwargs.get("trace_signal", cls.trace_signal)),
            trace_control=str(kwargs.get("trace_control", cls.trace_control)),
            trace_dropout_fraction=float(
                kwargs.get(
                    "trace_dropout_fraction", cls.trace_dropout_fraction
                )
            ),
            trace_noise_fraction=float(
                kwargs.get("trace_noise_fraction", cls.trace_noise_fraction)
            ),
            trace_edge_noise_fraction=float(
                kwargs.get(
                    "trace_edge_noise_fraction", cls.trace_edge_noise_fraction
                )
            ),
            frontier_abnormal_threshold=float(
                kwargs.get("frontier_abnormal_threshold", cls.frontier_abnormal_threshold)
            ),
            frontier_normal_threshold=float(
                kwargs.get("frontier_normal_threshold", cls.frontier_normal_threshold)
            ),
            frontier_edge_threshold=float(
                kwargs.get("frontier_edge_threshold", cls.frontier_edge_threshold)
            ),
            frontier_uncertain_stop_factor=float(
                kwargs.get(
                    "frontier_uncertain_stop_factor",
                    cls.frontier_uncertain_stop_factor,
                )
            ),
            trace_residual=bool(kwargs.get("trace_residual", cls.trace_residual)),
            trace_min_energy=float(kwargs.get("trace_min_energy", cls.trace_min_energy)),
            use_victim_demotion=bool(
                kwargs.get("use_victim_demotion", cls.use_victim_demotion)
            ),
            protect_coverage_parents=bool(
                kwargs.get(
                    "protect_coverage_parents",
                    cls.protect_coverage_parents,
                )
            ),
            victim_floor=float(kwargs.get("victim_floor", cls.victim_floor)),
            scorer=str(kwargs.get("scorer", cls.scorer)),
            cusum_slack=float(kwargs.get("cusum_slack", cls.cusum_slack)),
        )
