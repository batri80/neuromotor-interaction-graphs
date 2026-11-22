"""
Regenerates Figure 2 (population-pooled NIG visualization, PD vs HC)
using REAL computed TE matrices from your actual PaHaW data -- not
placeholder/synthetic data. This addresses R1.2/R2.7: shows genuine
population-level graph structure with quantitative edge-weight info
overlaid (addressing R2.7's "difficult to interpret quantitatively").

Run this locally where you have PaHaW data access.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from ngs_pipeline_v2 import load_pahaw, CONFIG, compute_te_matrix, build_nig, extract_ngs

CONFIG['pahaw_root']  = './PaHaW/PaHaW_public'
CONFIG['corpus_path'] = './PaHaW/PaHaW_files/corpus_PaHaW.xlsx'

part_df = load_pahaw(CONFIG['pahaw_root'], CONFIG['corpus_path'])
feat_cols = CONFIG['feature_names']
feat_labels = ['Eb','Eg','Em','sig_d','Ag','rho_s','Hs','Hg','SIA','Omega']

from sklearn.preprocessing import StandardScaler

X_pd = part_df[part_df['label']=='PD'][feat_cols].values
X_hc = part_df[part_df['label']=='HC'][feat_cols].values

Xz_pd = StandardScaler().fit_transform(X_pd)
Xz_hc = StandardScaler().fit_transform(X_hc)

G_pd = build_nig(Xz_pd, label='PD')
G_hc = build_nig(Xz_hc, label='HC')
ngs_pd = extract_ngs(G_pd)
ngs_hc = extract_ngs(G_hc)

NGS_CORE = ['H_graph','Eff','Modularity','Lambda2','SmallWorld','E_spectral','Ricci_OR']

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, G, title, ngs, color in [
    (axes[0], G_pd, 'PD (population-pooled, secondary formulation)', ngs_pd, '#C0392B'),
    (axes[1], G_hc, 'HC (population-pooled, secondary formulation)', ngs_hc, '#2874A6'),
]:
    pos = nx.circular_layout(G)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    max_w = max(weights) if weights else 1.0

    degrees = dict(G.degree(weight='weight'))
    node_sizes = [300 + 1200*(degrees.get(n,0)/max(degrees.values() or [1])) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                             node_color=color, alpha=0.85, edgecolors='black', linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax,
                              labels={i: feat_labels[i] for i in G.nodes()},
                              font_size=9, font_weight='bold')

    for (u, v, d) in G.edges(data=True):
        w = d['weight']
        lw = 0.5 + 4.5 * (w / max_w)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u,v)],
                                 width=lw, alpha=0.55, edge_color='gray',
                                 arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.08')

    ax.set_title(title, fontsize=11, weight='bold')
    ax.axis('off')

    # quantitative overlay -- addresses R2.7
    info = (f"n_edges={G.number_of_edges()}\n"
            f"H={ngs.get('H_graph',0):.3f}  Eff={ngs.get('Eff',0):.3f}\n"
            f"$\\sigma_{{SW}}$={ngs.get('SmallWorld',0):.3f}  "
            f"$E_{{spec}}$={ngs.get('E_spectral',0):.2f}\n"
            f"$\\bar\\kappa$={ngs.get('Ricci_OR',0):.3f}  Q={ngs.get('Modularity',0):.3f}")
    ax.text(0.5, -0.08, info, transform=ax.transAxes, ha='center', va='top',
             fontsize=8.5, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='gray'))

fig.suptitle('Population-Pooled Neuromotor Interaction Graphs (Secondary Formulation)\n'
             'Node size $\\propto$ weighted degree; edge width $\\propto$ TE weight',
             fontsize=12, y=1.02)

plt.tight_layout()
plt.savefig('Fig2_NIG_PD_vs_HC.pdf', bbox_inches='tight', dpi=300)
plt.savefig('Fig2_NIG_PD_vs_HC.png', bbox_inches='tight', dpi=200)
print("Saved Fig2_NIG_PD_vs_HC.pdf and .png with REAL computed data.")
print(f"\nPD NGS: {ngs_pd}")
print(f"HC NGS: {ngs_hc}")
