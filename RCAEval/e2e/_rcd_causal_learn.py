"""RCD's customized PC skeleton discovery on causal-learn's current API.

RCAEval historically installed RCD by symlinking the customized files under
``lib/causallearn`` over causal-learn 0.1.2.3.  This module is an API port of
those two skeleton-discovery functions so that RCD can share the project's
Python 3.10 / causal-learn 0.1.3.3 environment.  The edge-removal and
conditioning-set logic is intentionally kept identical to the bundled RCD
implementation.
"""

from itertools import combinations

import networkx as nx
import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.cit import CIT
from tqdm.auto import tqdm


class _RCDCausalGraph(CausalGraph):
    """Expose the bookkeeping used by RCD's causal-learn 0.1.2.3 patch."""

    def __init__(self, no_of_var, labels, data, indep_test):
        node_names = [labels.get(i, f"X{i + 1}") for i in range(no_of_var)]
        super().__init__(no_of_var, node_names=node_names)
        self._rcd_labels = dict(labels)
        self.set_ind_test(CIT(data, indep_test))
        self.p_values = np.empty((no_of_var, no_of_var), dtype=object)
        self.mi = np.empty(no_of_var, dtype=object)
        self._mi_index = 0
        self.no_ci_tests = 0

    def ci_test(self, i, j, conditioning_set):
        self.no_ci_tests += 1
        return super().ci_test(i, j, conditioning_set)

    def append_to_mi(self, node):
        self.mi[self._mi_index] = node
        self._mi_index += 1

    def remove_edge(self, x, y):
        edge = self.G.get_edge(self.G.nodes[x], self.G.nodes[y])
        if edge is not None:
            self.G.remove_edge(edge)
        reverse = self.G.get_edge(self.G.nodes[y], self.G.nodes[x])
        if reverse is not None:
            self.G.remove_edge(reverse)

    def to_nx_graph(self):
        """Preserve the old patch's service/metric labels in the NX graph."""
        labels = {
            i: self._rcd_labels.get(i, self.G.nodes[i].get_name())
            for i in range(len(self.G.graph))
        }
        graph = nx.DiGraph()
        graph.add_nodes_from(labels.values())
        for i, j in self.find_undirected():
            graph.add_edge(labels[i], labels[j], color="g")
        for i, j in self.find_fully_directed():
            graph.add_edge(labels[i], labels[j], color="b")
        for i, j in self.find_bi_directed():
            graph.add_edge(labels[i], labels[j], color="r")
        self.nx_graph = graph


def _new_graph(data, labels, indep_test):
    if not labels:
        labels = {i: f"X{i + 1}" for i in range(data.shape[1])}
    return _RCDCausalGraph(data.shape[1], labels, data, indep_test)


def skeleton_discovery(
    data,
    alpha,
    indep_test,
    stable=True,
    background_knowledge=None,
    labels=None,
    verbose=False,
    show_progress=True,
):
    """API-compatible port of ``lib/causallearn`` skeleton discovery."""
    assert type(data) is np.ndarray
    assert 0 < alpha < 1

    n_features = data.shape[1]
    cg = _new_graph(data, labels or {}, indep_test)
    depth = -1
    pbar = tqdm(total=n_features) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(n_features):
            if show_progress:
                pbar.update()
                pbar.set_description(f"Depth={depth}, working on node {x}")
            neigh_x = cg.neighbors(x)
            if len(neigh_x) < depth - 1:
                continue
            for y in neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                    background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                    and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])
                ):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        cg.remove_edge(x, y)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    edge_removal.extend(((x, y), (y, x)))

                neigh_x_without_y = np.delete(neigh_x, np.where(neigh_x == y))
                for conditioning_set in combinations(neigh_x_without_y, depth):
                    p_value = cg.ci_test(x, y, conditioning_set)
                    if p_value > alpha:
                        if verbose:
                            print(
                                "%d ind %d | %s with p-value %f\n"
                                % (x, y, conditioning_set, p_value)
                            )
                        if not stable:
                            cg.remove_edge(x, y)
                            append_value(cg.sepset, x, y, conditioning_set)
                            append_value(cg.sepset, y, x, conditioning_set)
                            break
                        edge_removal.extend(((x, y), (y, x)))
                        sepsets.update(conditioning_set)
                    else:
                        append_value(cg.p_values, x, y, p_value)
                        if verbose:
                            print(
                                "%d dep %d | %s with p-value %f\n"
                                % (x, y, conditioning_set, p_value)
                            )
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()
        for x, y in set(edge_removal):
            edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge is not None:
                cg.G.remove_edge(edge)

    if show_progress:
        pbar.close()
    return cg


def local_skeleton_discovery(
    data,
    local_node,
    alpha,
    indep_test,
    mi=None,
    labels=None,
    verbose=False,
):
    """API-compatible port of RCD's localized skeleton discovery."""
    assert type(data) is np.ndarray
    assert local_node <= data.shape[1]
    assert 0 < alpha < 1

    cg = _new_graph(data, labels or {}, indep_test)
    depth = -1
    x = local_node
    for node in mi or []:
        cg.remove_edge(x, node)

    while cg.max_degree() - 1 > depth:
        depth += 1
        local_neigh = np.random.permutation(cg.neighbors(x))
        for y in local_neigh:
            neigh_y = cg.neighbors(y)
            neigh_y = np.delete(neigh_y, np.where(neigh_y == x))
            neigh_y_f = []
            if depth > 0:
                neigh_y_f = [s for s in neigh_y if x in cg.neighbors(s)]

            for conditioning_set in combinations(neigh_y_f, depth):
                p_value = cg.ci_test(x, y, conditioning_set)
                if p_value > alpha:
                    if verbose:
                        print(
                            "%d ind %d | %s with p-value %f\n"
                            % (x, y, conditioning_set, p_value)
                        )
                    cg.remove_edge(x, y)
                    append_value(cg.sepset, x, y, conditioning_set)
                    append_value(cg.sepset, y, x, conditioning_set)
                    if depth == 0:
                        cg.append_to_mi(y)
                    break
                append_value(cg.p_values, x, y, p_value)
                if verbose:
                    print(
                        "%d dep %d | %s with p-value %f\n"
                        % (x, y, conditioning_set, p_value)
                    )
    return cg
