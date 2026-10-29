"""Generate report tables and vector figures from recorded measurements.

Use --refresh to copy the release results into this self-contained project.
Without --refresh, use the bundled data snapshots. No experiments are run.
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prepare_gate_ablation import build as build_gate_ablation

ROOT = Path(__file__).resolve().parent
COLORS = ['#0072B2', '#D55E00', '#009E73']
LABELS = ['Dense C', 'MoE', 'Dense P']


def table(filename, caption, label, columns, header, rows):
    text = [r'\begin{table}[ht]', r'\centering', r'\small',
            r'\caption{' + caption + '}', r'\label{' + label + '}',
            r'\begin{tabular}{' + columns + '}', r'\toprule',
            header + r' \\', r'\midrule']
    text += [row + r' \\' for row in rows]
    text += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (ROOT / 'generated' / filename).write_text('\n'.join(text) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    for name in ['data', 'generated', 'figures']:
        (ROOT / name).mkdir(exist_ok=True)
    if args.refresh:
        for name in ['numpy.json', 'language.json', 'run_provenance.json', 'gate_ablation.json']:
            shutil.copy2(ROOT.parents[1] / 'results' / name, ROOT / 'data' / name)
    numpy_log = json.loads((ROOT / 'data/numpy.json').read_text())
    log = json.loads((ROOT / 'data/language.json').read_text())
    modes = ['dense_compute', 'moe', 'dense_parameter']
    runs = [next(r for r in log['runs'] if r['mode'] == mode and r['seed'] == 416)
            for mode in modes]
    routing = next(r for r in log['routing_control'] if r['seed'] == 416)
    checks = numpy_log['checks']
    names = {'W1': 'W_1', 'b1': 'b_1', 'W2': 'W_2', 'b2': 'b_2',
             'G': 'G', 'C': 'C', 'bc': 'b_c', 'input': 'x'}
    rows = []
    for key, name in names.items():
        values = [checks[c]['max_normalized_errors'][key]
                  for c in ['handwritten_toy', 'sparse_top2']]
        rows.append(f'${name}$ & {values[0]:.3e} & {values[1]:.3e}')
    table('numpy_gradients.tex', 'Maximum normalized central-difference errors.',
          'tab:numpy-gradients', 'lrr', 'Quantity & Two-expert example & Three-expert Top-2', rows)
    rows = []
    for pred in numpy_log['training']['predictions']:
        context = ''.join(chr(b) for b in pred['context_bytes']).replace(' ', r'\textvisiblespace{}')
        def byte_label(b):
            return 'space (32)' if b == 32 else f'{chr(b)} ({b})'
        rows.append(r'\texttt{' + context + '} & ' + byte_label(pred['target_byte']) +
                    ' & ' + byte_label(pred['predicted_byte']))
    table('numpy_predictions.tex', 'Three predictions after 400 NumPy SGD updates. Parentheses contain byte values.',
          'tab:numpy-predictions', 'lll', 'Four-byte context & Target & Prediction', rows)
    rows = []
    for run, label in zip(runs, LABELS):
        t = run['test']
        rows.append(f"{label} & {run['parameters']:,} & {run['matrix_macs_per_token']:,} & "
                    f"{t['domain_ce'][0]:.4f} & {t['domain_ce'][1]:.4f} & "
                    f"{t['ce']:.4f} & {t['perplexity']:.4f} & {run['forward_batch_median_ms']:.3f}")
    table('language_results.tex',
          'Test results and costs. C/P denote computation/parameter matching; latency is in milliseconds per full batch.',
          'tab:language-results', '@{}lrrrrrrr@{}',
          'Model & Params & MACs/token & Text CE & Code CE & Mean CE & PPL & ms', rows)

    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(4, 1.8), layout='constrained')
    history = numpy_log['training']['history']
    ax.plot([h['step'] for h in history], [h['ce'] for h in history], color=COLORS[0])
    ax.set(xlabel='SGD update', ylabel='Training CE (nats)')
    ax.grid(alpha=.2)
    fig.savefig(ROOT / 'figures/numpy_loss.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), layout='constrained')
    for run, label, color, marker in zip(runs, LABELS, COLORS, ['o', 's', '^']):
        history = run['history']
        steps = [h['step'] for h in history]
        axes[0].plot(steps, [h['train_ce'] for h in history], label=label,
                     color=color, marker=marker, markersize=3, linewidth=1)
        axes[1].plot(steps, [h['validation']['ce'] for h in history], label=label,
                     color=color, marker=marker, markersize=3, linewidth=1)
    for ax, ylabel in zip(axes, ['Training CE (nats)', 'Validation CE (nats)']):
        ax.set(xlabel='Update', ylabel=ylabel, xticks=[500, 1500, 3000])
        ax.grid(alpha=.2)
    axes[1].legend(frameon=False)
    fig.savefig(ROOT / 'figures/language_curves.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.8), layout='constrained')
    ax = axes[0, 0]
    for run, label, color in zip(runs, LABELS, COLORS):
        ax.scatter(run['matrix_macs_per_token']/1000, run['test']['ce'],
                   s=run['parameters']/4000, label=label, color=color)
    ax.set(xlabel='MACs/token (thousands)', ylabel='Test CE (nats)')
    ax.text(.95, .95, '(a)', transform=ax.transAxes, va='top', ha='right')
    ax.legend(loc='center right', frameon=False, markerscale=.65)
    ax.grid(alpha=.2)
    ax = axes[0, 1]
    bars = ax.bar(LABELS, [r['forward_batch_median_ms'] for r in runs], color=COLORS)
    ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=7)
    ax.set(ylabel='Forward latency (ms)', ylim=(0, 10))
    ax.text(.04, .95, '(b)', transform=ax.transAxes, va='top')
    ax = axes[1, 0]
    normal, mean = routing['normal_test']['ce'], routing['diagnostic_ce_mean']
    low, high = routing['diagnostic_ce_range']
    bars = ax.bar(['Normal', 'Perturbed'], [normal, mean], color=[COLORS[1], '#CC79A7'])
    ax.errorbar([1], [mean], yerr=[[mean-low], [high-mean]], fmt='none', color='black', capsize=4)
    ax.bar_label(bars, fmt='%.3f', padding=5, fontsize=7)
    ax.set(ylabel='CE / diagnostic CE (nats)', ylim=(0, 3.6))
    ax.text(.04, .95, '(c)', transform=ax.transAxes, va='top')
    ax = axes[1, 1]
    usage = np.asarray(routing['normal_test']['route_fraction']).reshape(4, 16)
    heat = ax.imshow(usage, aspect='auto', cmap='Blues', vmin=0, vmax=.15)
    ax.set(xlabel='Expert index', xticks=[0, 3, 6, 9, 12, 15], yticks=range(4),
           yticklabels=['Text L1', 'Text L2', 'Code L1', 'Code L2'])
    ax.text(.04, .95, '(d)', transform=ax.transAxes, va='top', color='black')
    fig.colorbar(heat, ax=ax, orientation='horizontal', pad=.13,
                 ticks=[0, .05, .10, .15], label='Selection fraction', fraction=.07)
    fig.savefig(ROOT / 'figures/task2_analysis.pdf')
    plt.close(fig)
    build_gate_ablation(ROOT)
    print('Prepared tables and vector figures from saved results.')


if __name__ == '__main__':
    main()
