"""
Redraw Figure 2 (population-pooled PD and HC NIGs, PaHaW) with
  * all seven NGS values in the overlay (lambda_2 was missing),
  * Table 1 symbols as node labels instead of code names,
  * text large enough to read at print size.
Graph construction is Algorithm 1 of the manuscript, unchanged
(ngs_pipeline_v2.build_nig / extract_ngs on group-standardised Phi10).
The small-world null graphs are seeded, so the values are reproducible.

Usage:  python r2_fig2_redraw.py [--pahaw ./PaHaW/PaHaW_public]
                                 [--corpus ./PaHaW/PaHaW_files/corpus_PaHaW.xlsx]
Output: Fig2_NIG_PD_vs_HC.pdf, Fig2_NIG_PD_vs_HC.png, r2_fig2_values.txt
"""
import argparse, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
import ngs_pipeline_v2 as base
from ngs_pipeline_v2 import CONFIG, build_nig, extract_ngs

LABELS = dict(base.FEAT_LABELS)          # same symbols as Table 1
LABELS['SIA'] = r'$\mathrm{SIA}$'

ap = argparse.ArgumentParser()
ap.add_argument('--pahaw', default='./PaHaW/PaHaW_public')
ap.add_argument('--corpus', default='./PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
args = ap.parse_args()

df = base.load_pahaw(args.pahaw, args.corpus)
feat = CONFIG['feature_names']
graphs, vals = {}, {}
for g in ['PD', 'HC']:
    X = StandardScaler().fit_transform(df[df.label == g][feat].values.astype(float))
    G = build_nig(X, label=g)
    random.seed(0); np.random.seed(0)
    graphs[g], vals[g] = G, extract_ngs(G)

with open('r2_fig2_values.txt', 'w') as f:
    for g in ['PD', 'HC']:
        v = vals[g]
        line = (f"{g}: n={int((df.label==g).sum())} edges={v['n_edges']} H={v['H_graph']:.3f} "
                f"Eff={v['Eff']:.3f} Q={v['Modularity']:.3f} lambda2={v['Lambda2']:.3f} "
                f"SW={v['SmallWorld']:.3f} Espec={v['E_spectral']:.2f} kappa={v['Ricci_OR']:.3f}")
        print(line); f.write(line + '\n')

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})
fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
pos = nx.circular_layout(graphs['PD'])
wmax = max(max((d['weight'] for _, _, d in G.edges(data=True)), default=1) for G in graphs.values())
for ax, g, col in zip(axes, ['PD', 'HC'], ['#c0504d', '#4f81bd']):
    G, v = graphs[g], vals[g]
    deg = dict(G.degree(weight='weight'))
    dmax = max(deg.values()) or 1
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, edge_color='0.55', alpha=0.6,
                           width=[0.4 + 2.6 * d['weight'] / wmax for _, _, d in G.edges(data=True)])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=col, edgecolors='k', linewidths=0.6,
                           node_size=[350 + 650 * deg[n] / dmax for n in G.nodes()])
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color='white',
                            labels={i: LABELS[feat[i]] for i in G.nodes()})
    ax.set_title(f"{g} (population-pooled, $n={int((df.label==g).sum())}$)", fontsize=13)
    txt = (f"edges $={v['n_edges']}$    $H={v['H_graph']:.3f}$    Eff $={v['Eff']:.3f}$\n"
           f"$Q={v['Modularity']:.3f}$    $\\lambda_2={v['Lambda2']:.3f}$    "
           f"$\\sigma_{{SW}}={v['SmallWorld']:.3f}$\n"
           f"$E_{{spec}}={v['E_spectral']:.1f}$    $\\bar\\kappa={v['Ricci_OR']:.3f}$")
    ax.text(0.5, -0.06, txt, transform=ax.transAxes, ha='center', va='top', fontsize=12,
            bbox=dict(boxstyle='round', fc='0.96', ec='0.7'))
    ax.set_axis_off(); ax.set_aspect('equal')
fig.tight_layout()
fig.savefig('Fig2_NIG_PD_vs_HC.pdf', bbox_inches='tight')
fig.savefig('Fig2_NIG_PD_vs_HC.png', dpi=300, bbox_inches='tight')
print("Saved Fig2_NIG_PD_vs_HC.pdf/.png and r2_fig2_values.txt")
