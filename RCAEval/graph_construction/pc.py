from math import sqrt

import numpy as np
from scipy.stats import norm

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import FisherZ, chisq, fisherz, gsq, kci, mv_fisherz
from causallearn.utils.PCUtils import Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

background_knowledge = BackgroundKnowledge()
background_knowledge.add_forbidden_by_pattern(".*mem$", ".*lat50$")
background_knowledge.add_forbidden_by_pattern(".*cpu$", ".*lat50$")
background_knowledge.add_forbidden_by_pattern(".*", "frontend.*")


def pc_default(data, show_progress=False, with_bg=False, **kwargs):
    node_names = data.columns.to_list()

    cg = pc(
        data.to_numpy().astype(float),
        node_names=node_names,
        show_progress=show_progress,
        background_knowledge=background_knowledge if with_bg else None,
    )
    return cg.G.graph


class _NumericallyStableFisherZ(FisherZ):
    """Fisher-Z with deterministic handling of singular telemetry matrices.

    RE2 contains distinct metrics that can be perfectly collinear.  The
    causal-learn implementation directly inverts each correlation submatrix;
    at the floating-point boundary this can produce ``abs(r) > 1`` and make
    the Fisher transform fail with a math-domain error.  A Moore-Penrose
    inverse and a one-ULP clip preserve the limiting test while keeping all
    declared telemetry variables in the candidate set.
    """

    def __call__(self, X, Y, condition_set=None):
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(
            X, Y, condition_set
        )
        if cache_key in self.pvalue_cache:
            return self.pvalue_cache[cache_key]

        variables = Xs + Ys + condition_set
        correlation = self.correlation_matrix[np.ix_(variables, variables)]
        def partial_correlation_from(inverse):
            denominator_squared = float(inverse[0, 0] * inverse[1, 1])
            if not np.isfinite(denominator_squared) or denominator_squared <= 0:
                return None
            value = -float(inverse[0, 1]) / sqrt(denominator_squared)
            return value if np.isfinite(value) else None

        try:
            inverse = np.linalg.inv(correlation)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(correlation, hermitian=True)

        partial_correlation = partial_correlation_from(inverse)
        if partial_correlation is None or abs(partial_correlation) >= 1.0:
            inverse = np.linalg.pinv(correlation, hermitian=True)
            partial_correlation = partial_correlation_from(inverse)
        if partial_correlation is None:
            # A variable has zero conditional variance.  The CI relation is
            # not identifiable, so conservatively retain the edge instead of
            # claiming independence or choosing an arbitrary ridge constant.
            self.pvalue_cache[cache_key] = 0.0
            return 0.0
        limit = 1.0 - np.finfo(float).eps
        partial_correlation = float(
            np.clip(partial_correlation, -limit, limit)
        )
        statistic = sqrt(self.sample_size - len(condition_set) - 3) * abs(
            np.arctanh(partial_correlation)
        )
        p_value = float(2 * norm.sf(statistic))
        self.pvalue_cache[cache_key] = p_value
        return p_value


def pc_numerically_stable(data, show_progress=False, with_bg=False):
    """Run default PC with a numerically stable Fisher-Z implementation."""
    node_names = data.columns.to_list()
    values = data.to_numpy().astype(float)
    independence_test = _NumericallyStableFisherZ(values)
    skeleton = SkeletonDiscovery.skeleton_discovery(
        values,
        alpha=0.05,
        indep_test=independence_test,
        stable=True,
        background_knowledge=background_knowledge if with_bg else None,
        show_progress=show_progress,
        node_names=node_names,
    )
    oriented = UCSepset.uc_sepset(
        skeleton,
        priority=2,
        background_knowledge=background_knowledge if with_bg else None,
    )
    return Meek.meek(
        oriented,
        background_knowledge=background_knowledge if with_bg else None,
    )


def pc_fisherz(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=fisherz,
        stable=False,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pc_fisherz_stable(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=fisherz,
        stable=True,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pc_gsq(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=gsq,
        stable=False,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pc_gsq_stable(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=gsq,
        stable=True,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pc_chisq(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=chisq,
        stable=False,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pc_chisq_stable(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=chisq,
        stable=True,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pcmci(data):
    raise NotImplementedError
