"""Figure 1: NIG/NGS construction, primary (within-subject) vs secondary
(population-pooled) formulation. Redrawn for the revision so that the
secondary pipeline's role (descriptive visualisation and leakage audit
only) matches the text, and the table reference is removed."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, BLUE_F = '#1f5a8a', '#eaf2fb'
GREY, GREY_F = '#6b6b6b', '#f2f2f2'

fig, ax = plt.subplots(figsize=(20, 11.3))
ax.set_xlim(0, 20); ax.set_ylim(0, 11.3); ax.axis('off')

def box(x, y, w, h, text, edge, face, bold_lines=(), fs=14.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                linewidth=2.2, edgecolor=edge, facecolor=face))
    lines = text.split('\n')
    n = len(lines)
    for k, line in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 + (n - 1) * 0.14 - k * 0.28, line, ha='center',
                va='center', fontsize=fs,
                fontweight='bold' if k in bold_lines else 'normal')

def arrow(x0, x1, y, col):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='-|>', color=col, lw=2.2, mutation_scale=18))

ax.text(10, 10.85, 'NIG/NGS Construction: Primary vs. Secondary Formulation',
        ha='center', va='center', fontsize=22, fontweight='bold')

# Primary row
W, H, Y1 = 2.8, 1.65, 8.15
xs = [0.9, 3.95, 7.0, 10.05, 13.1, 16.15]
prim = [('Raw SVC stream\n$(y,x,t,b,\\theta,\\phi,p)$\nper participant', ()),
        ('Chronological\ntemporal windows\n$W=20$, equal-sample', ()),
        ('$\\Phi^{10}_{\\mathrm{kin}}$\nper window\n(10 kinematic feats.)', ()),
        ('Within-subject TE\n$\\ell=1$, $B=8$ bins\n$\\tau_s=Q_{0.25}$', ()),
        ('NIG$_s$\n(per-participant\ngraph, 10 nodes)', (1, 2)),
        ('NGS$_s$\n(7-dim signature)', (1,))]
for x, (t, b) in zip(xs, prim):
    box(x, Y1, W, H, t, BLUE, BLUE_F, b)
for k in range(5):
    arrow(xs[k] + W, xs[k + 1], Y1 + H / 2, BLUE)
ax.text(2.3, 10.2, "No other participant's data or label enters this construction "
        "(leakage-free by design)", color=BLUE, fontsize=13, style='italic', va='center')
ax.text(0.45, 9.3, 'PRIMARY', rotation=90, color=BLUE, fontsize=17, fontweight='bold',
        ha='center', va='center')
box(9.3, 6.25, 7.4, 1.4,
    'Nested fold-safe feature selection $\\rightarrow$ LOPO-CV / repeated $k$-fold\n'
    'AUC $=0.6150$ (LOPO), $p=0.0554$, 95% CI $[0.485,0.740]$\n'
    'Estimator bias and reliability analysis: Section 5.13, Appendix B',
    BLUE, BLUE_F, fs=13)
ax.plot([16.4, 14.1], [Y1, 7.65], color=BLUE, lw=2.0)

ax.plot([0.6, 19.4], [5.1, 5.1], ls='--', color='#aaaaaa', lw=1.5)

# Secondary row
Y2 = 2.4
sec = [('Handwriting\nimages, ALL\nparticipants', ()),
       ('$\\Phi^{10}$\nimage features\n(all subjects)', ()),
       ('Population-pooled\nTE across\nparticipants', ()),
       ('NIG$_G$\n(group graph,\nPD or HC)', (1, 2)),
       ('Descriptive\nvisualisation (Fig. 2)\nand leakage audit only', (0, 1, 2))]
for k, (x, (t, b)) in enumerate(zip(xs[:5], sec)):
    box(x, Y2, W, H, t, GREY, GREY_F, b, fs=13 if k == 4 else 14.5)
for k in range(4):
    arrow(xs[k] + W, xs[k + 1], Y2 + H / 2, GREY)
ax.text(0.45, 3.2, 'SECONDARY', rotation=90, color=GREY, fontsize=17, fontweight='bold',
        ha='center', va='center')
ax.text(2.3, 2.05, 'NOT used for group comparisons or individual-level classification: '
        'pooled graph estimation permits', color=GREY, fontsize=12.5, style='italic', va='center')
ax.text(2.3, 1.78, 'evaluation-participant information to influence the representation '
        'used to classify them', color=GREY, fontsize=12.5, style='italic', va='center')

# Legend
for y, edge, face, lab in [(0.62, BLUE, BLUE_F, 'Primary formulation (used for all group comparisons '
                                                 'and individual-level inference)'),
                           (0.30, GREY, GREY_F, 'Secondary formulation (descriptive visualisation '
                                                'and leakage audit only)')]:
    ax.add_patch(FancyBboxPatch((6.25, y - 0.08), 0.4, 0.16, boxstyle='square,pad=0',
                                linewidth=1.5, edgecolor=edge, facecolor=face))
    ax.text(6.85, y, lab, fontsize=13.5, va='center')

fig.savefig('manuscript/Fig1_framework.pdf', bbox_inches='tight')
fig.savefig('manuscript/Fig1_framework.png', dpi=130, bbox_inches='tight')
print('saved')
