"""
Sensitivity of the primary within-subject results (W=20, B=8, lag 1, 25th-percentile
pruning) to two implementation conventions of the undirected NGS descriptors:

  1. Directed-to-undirected conversion. networkx DiGraph.to_undirected() keeps, for a
     reciprocal pair i<->j, the weight of the edge added last, i.e. the edge from the
     higher-indexed to the lower-indexed node ("last", the released convention).
     Alternatives: the mean and the maximum of the two directed weights.
  2. Curvature over at most the first 40 undirected edges (released) versus all edges.

Graph entropy and spectral energy use the directed weights and are unaffected by (1).
For each variant the script reports: Spearman correlation of each descriptor with the
released version, the PD-HC Mann-Whitney comparison with BH-FDR (Table 5), and the
nested LOPO-AUC of the primary classifier (Section 5.2; released value 0.6150).

Usage:  python r2_undirected_rule.py [--pahaw ...] [--corpus ...]
Output: r2_undirected_rule_results.txt
"""
import argparse, random
import numpy as np
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
import r2_common as rc
import ngs_pipeline_v2 as base
from ngs_pipeline_v2 import NGS_CORE, CONFIG

ap = argparse.ArgumentParser()
ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
a = ap.parse_args()


def reweighted(G, rule):
    """DiGraph whose reciprocal pairs carry the aggregated weight in both directions,
    so that to_undirected() returns that weight whatever the edge order."""
    H = G.copy()
    for u, v in list(G.edges()):
        if G.has_edge(v, u):
            w1, w2 = G[u][v]['weight'], G[v][u]['weight']
            H[u][v]['weight'] = (w1 + w2) / 2 if rule == 'mean' else max(w1, w2)
    return H


def ricci_all_edges(G):
    """ngs_pipeline_v2.ollivier_ricci without the 40-edge cap."""
    alpha = CONFIG['ricci_alpha']
    U = G.to_undirected()
    if U.number_of_edges() == 0:
        return 0.0
    def mu(v):
        m = {v: alpha}; nb = list(U.neighbors(v))
        if not nb: return m
        wt = sum(U[v][u].get('weight', 1.0) for u in nb)
        for u in nb: m[u] = (1 - alpha) * U[v][u].get('weight', 1.0) / (wt + 1e-10)
        return m
    def cost(m1, m2):
        t = 0.0
        for x, px in m1.items():
            for z, pz in m2.items():
                try: d = nx.shortest_path_length(U, x, z)
                except nx.NetworkXNoPath: d = len(U)
                t += px * pz * d
        return t
    ks = [1.0 - cost(mu(u), mu(v)) / U[u][v]['weight']
          for u, v in U.edges() if U[u][v].get('weight', 1.0) >= 1e-10]
    return float(np.mean(ks)) if ks else 0.0


def ngs(G):
    random.seed(0)
    d = base.extract_ngs(G)
    return [d.get(k, 0.0) for k in NGS_CORE]


ids, y, mats, _ = rc.load_cohort(a.pahaw, a.corpus)
y = np.array(y)
V = {'released': [], 'mean': [], 'max': [], 'ricci_all': []}
n_pairs, n_capped = [], 0
ki = NGS_CORE.index('Ricci_OR')
for Xw in mats:
    M = rc.te_matrix_codes(rc.discretise_columns(StandardScaler().fit_transform(Xw)))
    G = rc.graph_from_matrix(M)
    base_vals = ngs(G)
    V['released'].append(base_vals)
    directed_only = [NGS_CORE.index('H_graph'), NGS_CORE.index('E_spectral')]
    for rule in ['mean', 'max']:
        vals = ngs(reweighted(G, rule))
        for k in directed_only:          # these use the directed weights: unchanged
            vals[k] = base_vals[k]
        V[rule].append(vals)
    r = list(base_vals); r[ki] = ricci_all_edges(G); V['ricci_all'].append(r)
    m = G.to_undirected().number_of_edges(); n_pairs.append(m); n_capped += m > 40
V = {k: np.array(v) for k, v in V.items()}

f = open('r2_undirected_rule_results.txt', 'w')
def out(s=''): print(s); f.write(s + '\n')
out(f"n={len(y)}  undirected pairs per graph: median {int(np.median(n_pairs))}, "
    f"range {min(n_pairs)}-{max(n_pairs)}; graphs with >40 pairs (curvature capped): {n_capped}")
for name in ['released', 'mean', 'max', 'ricci_all']:
    X = V[name]
    auc = rc.nested_lopo_auc(X, y)
    out(f"\n== {name}   nested LOPO-AUC = {auc:.4f}")
    out(f"{'descriptor':<12}{'rho vs released':>16}{'MW p':>9}{'BH q':>8}")
    for k, (m, p, q, _) in enumerate(rc.group_comparison(X, y)):
        rho = stats.spearmanr(X[:, k], V['released'][:, k])[0] if np.std(X[:, k]) > 0 else float('nan')
        out(f"{m:<12}{rho:>16.3f}{p:>9.4f}{q:>8.4f}")
f.close()
