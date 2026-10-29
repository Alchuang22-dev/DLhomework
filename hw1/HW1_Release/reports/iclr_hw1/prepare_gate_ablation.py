"""Build the Task 3.4.3 table and figure from the bundled evaluation snapshot.

Use --refresh to first copy results/gate_ablation.json from the release directory.
No model evaluation or training is performed by this script.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
LABELS = {'original': 'Original', 'uniform': 'Uniform', 'midpoint': 'Midpoint', 'swapped': 'Swapped'}


def build(root=ROOT):
    data = json.loads((root / 'data/gate_ablation.json').read_text())
    conditions = data['conditions']
    for directory in ['generated', 'figures']:
        (root / directory).mkdir(exist_ok=True)
    rows = []
    table = [r'\begin{table}[ht]', r'\centering', r'\small',
             r'\caption{Last-layer weight intervention on the fixed seed-416 checkpoint. '
             r'CE is in nats; $\Delta$ is relative to original mean CE.}',
             r'\label{tab:gate-ablation}', r'\begin{tabular}{@{}lrrrrrr@{}}',
             r'\toprule', r'Weights & $\lambda$ & Text CE & Code CE & Mean CE & $\Delta$ CE & PPL \\',
             r'\midrule']
    for condition in conditions:
        test = condition['test']
        name = LABELS[condition['condition']]
        table.append(f"{name} & {condition['lambda']:g} & {test['domain_ce'][0]:.4f} & "
                     f"{test['domain_ce'][1]:.4f} & {test['ce']:.4f} & "
                     f"{condition['delta_ce']:+.4f} & {test['perplexity']:.4f}" + r' \\')
        rows.append({'condition': condition['condition'], 'lambda': condition['lambda'],
                     'text_ce': test['domain_ce'][0], 'code_ce': test['domain_ce'][1],
                     'mean_ce': test['ce'], 'delta_ce': condition['delta_ce'],
                     'ppl': test['perplexity'],
                     'text_delta_ce': condition['domain_delta_ce'][0],
                     'code_delta_ce': condition['domain_delta_ce'][1]})
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (root / 'generated/gate_ablation.tex').write_text('\n'.join(table) + '\n')
    with (root / 'generated/gate_ablation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), layout='constrained')
    ordered = sorted(conditions, key=lambda r: r['lambda'])
    xs = [r['lambda'] for r in ordered]
    ax = axes[0]
    for domain, color, marker in [(0, '#0072B2', 'o'), (1, '#D55E00', 's')]:
        ax.plot(xs, [r['domain_delta_ce'][domain] for r in ordered], marker=marker,
                color=color, markersize=3, linewidth=1,
                label=['Text', 'Code'][domain])
    ax.axhline(0, color='gray', linewidth=.6, linestyle='--')
    ax.set(xlabel=r'Weight interpolation $\lambda$', ylabel=r'$\Delta$ CE from original (nats)',
           xticks=[-1, 0, .5, 1])
    ax.text(.42, .95, '(a)', transform=ax.transAxes, va='top')
    ax.legend(loc='upper right', frameon=False)
    ax.grid(alpha=.15)

    ax = axes[1]
    edges = np.asarray(data['gap_histogram_edges'])
    for stat, color, linestyle, label in zip(data['original_gate_gap_statistics'],
                                            ['#0072B2', '#D55E00'], ['-', '--'], ['Text', 'Code']):
        fractions = np.asarray(stat['histogram_counts']) / stat['tokens']
        ax.stairs(fractions, edges, label=label, color=color, linestyle=linestyle, linewidth=1.2)
    ax.set(xlabel=r'Original gate gap $|g_1-g_2|$', ylabel='Fraction of domain tokens',
           xlim=(0, 1))
    ax.text(.04, .95, '(b)', transform=ax.transAxes, va='top')
    ax.legend(loc='upper left', bbox_to_anchor=(0, .88), frameon=False)
    ax.grid(alpha=.15)
    fig.savefig(root / 'figures/gate_ablation.pdf')
    fig.savefig(root / 'figures/gate_ablation.png', dpi=250)
    plt.close(fig)
    print('Wrote Task 3.4.3 comparison table, CSV, and vector/raster figure.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    if args.refresh:
        shutil.copy2(ROOT.parents[1] / 'results/gate_ablation.json', ROOT / 'data/gate_ablation.json')
    build()


if __name__ == '__main__':
    main()
