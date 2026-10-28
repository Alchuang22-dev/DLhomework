"""Build Task 1/2 figures and a comparison table from the official result log.

Run after train.py --mode all and diagnostics.py, using the same results folder.
This script reads results only; it does not train or alter evaluation settings.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    'dense_compute': 'Dense (compute matched)',
    'moe': 'Sparse MoE',
    'dense_parameter': 'Dense (parameter matched)',
}
COLORS = ['#3973ac', '#dd7c28', '#39866f']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results')
    args = parser.parse_args()
    out = args.results
    log = json.loads((out / 'language.json').read_text())
    runs = [next(r for r in log['runs'] if r['mode'] == mode and r['seed'] == 416)
            for mode in LABELS]
    routing = next(r for r in log['routing_control'] if r['seed'] == 416)
    labels = list(LABELS.values())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})

    # The supplied loop records current-batch training CE every 500 steps.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for run, label, color in zip(runs, labels, COLORS):
        history = run['history']
        steps = [h['step'] for h in history]
        axes[0].plot(steps, [h['train_ce'] for h in history], 'o-',
                     label=label, color=color)
        axes[1].plot(steps, [h['validation']['ce'] for h in history], 'o-',
                     label=label, color=color)
    for ax, title in zip(axes, ['Training CE (logged batches)', 'Validation CE']):
        ax.set(title=title, xlabel='Optimizer update', ylabel='Cross-entropy (nats)')
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.savefig(out / 'task1_curves.png', dpi=200)
    plt.close(fig)

    # One composite figure for Task 2: quality/cost, routing, and expert usage.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    ax = axes[0, 0]
    for run, label, color in zip(runs, labels, COLORS):
        ax.scatter(run['matrix_macs_per_token'] / 1000, run['test']['ce'],
                   s=run['parameters'] / 1600, color=color, label=label,
                   edgecolors='white', linewidths=1)
    ax.set(title='A. Quality and arithmetic cost', xlabel='Matrix MACs per token (thousands)',
           ylabel='Test CE (equal domain weighting)')
    ax.legend(fontsize=8)
    ax.text(.98, .98, 'Marker area represents parameter count',
            transform=ax.transAxes, ha='right', va='top', fontsize=8)
    ax.grid(alpha=.2)

    ax = axes[0, 1]
    short_labels = ['Dense\ncompute', 'MoE', 'Dense\nparameter']
    bars = ax.bar(short_labels, [r['forward_batch_median_ms'] for r in runs], color=COLORS)
    ax.bar_label(bars, fmt='%.2f ms', padding=4, fontsize=9)
    ax.set(title='B. Full-batch forward latency', ylabel='Median latency (ms)')
    ax.margins(y=.2)
    ax.text(.02, .96, 'CPU; B=32, T=64; includes dispatch',
            transform=ax.transAxes, va='top', fontsize=8)

    ax = axes[1, 0]
    normal = routing['normal_test']['ce']
    mean = routing['diagnostic_ce_mean']
    low, high = routing['diagnostic_ce_range']
    ax.bar([0, 1], [normal, mean], color=[COLORS[1], '#7b6ea8'])
    ax.errorbar([1], [mean], yerr=[[mean - low], [high - mean]],
                fmt='none', color='black', capsize=6)
    ax.scatter([1] * 3, [t['diagnostic']['ce'] for t in routing['position_wise_trials']],
               color='black', s=18, zorder=3)
    ax.set(xticks=[0, 1], xticklabels=['Normal test', 'Paired-route perturbation'],
           ylabel='Cross-entropy (nats)', title='C. Last-layer routing diagnostic')
    ax.text(.02, .97, '3 fixed seeds; whiskers show min/max\nPerturbation is batch-dependent, not LM performance',
            transform=ax.transAxes, va='top', fontsize=8)
    ax.margins(y=.3)

    ax = axes[1, 1]
    usage = np.asarray(routing['normal_test']['route_fraction'])
    heat = ax.imshow(usage.reshape(4, -1), aspect='auto', cmap='Blues', vmin=0)
    ax.set(title='D. Expert selection fractions by domain/layer', xlabel='Expert index',
           yticks=range(4), yticklabels=['Text / layer 1', 'Text / layer 2',
                                       'Code / layer 1', 'Code / layer 2'],
           xticks=range(usage.shape[-1]))
    fig.colorbar(heat, ax=ax, label='Fraction of selected token–expert pairs', shrink=.8)
    fig.savefig(out / 'task2_comparison.png', dpi=200)
    plt.close(fig)

    rows = []
    for run, label in zip(runs, labels):
        rows.append(dict(model=label, parameters=run['parameters'],
                         macs_per_token=run['matrix_macs_per_token'],
                         text_ce=run['test']['domain_ce'][0],
                         code_ce=run['test']['domain_ce'][1],
                         test_ce=run['test']['ce'], test_ppl=run['test']['perplexity'],
                         forward_batch_ms=run['forward_batch_median_ms'],
                         best_validation_ce=run['best_validation_ce'],
                         train_validation_seconds=run['train_validation_seconds']))
    with (out / 'task2_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = [
        '# Part Three: Task 1 and Task 2 — measured results', '',
        'Seed 416; 3000 updates per model; batch 32; context 64; CPU; 4 threads. '
        'The supplied optimizer, schedule, sampling order, and validation-checkpoint rule were retained. '
        'Test evaluation uses disjoint full-length packs and equal domain weighting.', '',
        '## Task 1', '',
        'The official dispatch/gradient and prefix-invariance checks passed. '
        'Training CE below is the current batch at each logging point, not an epoch average.', '',
        '![Training and validation curves](task1_curves.png)', '',
        '## Task 2', '',
        '| Model | Parameters | MACs/token | Text CE | Code CE | Test CE | Test PPL | Forward ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        report.append(f"| {row['model']} | {row['parameters']:,} | {row['macs_per_token']:,} | "
                      f"{row['text_ce']:.4f} | {row['code_ce']:.4f} | {row['test_ce']:.4f} | "
                      f"{row['test_ppl']:.4f} | {row['forward_batch_ms']:.3f} |")
    compute, moe, parameter = rows
    report.extend([
        '', '![Quality, cost, routing, and usage](task2_comparison.png)', '',
        f"MoE has {moe['parameters']/compute['parameters']:.2f} times the compute-matched dense model's "
        f"parameters at the same {moe['macs_per_token']:,} matrix MACs/token. "
        f"Its measured forward latency is {moe['forward_batch_ms']/compute['forward_batch_ms']:.2f} times as large. "
        'Arithmetic counts do not capture dispatch, gather/scatter, and small matrix-operation overheads.', '',
        f"MoE minus compute-matched dense test CE: {moe['test_ce']-compute['test_ce']:+.4f} nats. "
        f"MoE minus parameter-matched dense test CE: {moe['test_ce']-parameter['test_ce']:+.4f} nats. "
        f"The dense parameter match differs from MoE by "
        f"{100*abs(parameter['parameters']-moe['parameters'])/moe['parameters']:.3f}%.", '',
        f'Normal MoE test CE: {normal:.6f}. Paired-route diagnostic CE mean: {mean:.6f}; '
        f'range across three seeds: [{low:.6f}, {high:.6f}]; difference: {mean-normal:+.6f}. '
        'Expert IDs and gates are exchanged jointly across sequences at the same position in the last layer. '
        'Expert–token counts are unchanged; both prefix-invariance checks passed. '
        'This is a batch-dependent checkpoint diagnostic, not standalone LM performance or a causal estimate '
        'of the benefit of routing during training.', '',
        'Usage fractions show selection patterns across domains and layers; usage alone does not establish '
        'semantic specialization. These comparisons use one model seed and fixed updates/data order, '
        'not matched wall-clock time or convergence. Latencies are medians of 30 full-batch forwards '
        'after 10 warmups on the same CPU and include dispatch.', '',
        '## Reproduction and provenance', '',
        'Raw results: `language.json`; console logs: `training.log`, `diagnostics.log`; '
        'source hashes/environment and initial checks: `run_provenance.json`. '
        'The selected MoE checkpoint is `moe_seed416.pt`.', '',
        'Run `python MoE/summarize_results.py` from the release directory to recreate this report and figures. '
        'Review these measured results and disclose AI assistance when incorporating them into your submission.',
    ])
    (out / 'task1_task2_report.md').write_text('\n'.join(report) + '\n')
    print(f'Wrote figures, comparison CSV, and report to {out}')


if __name__ == '__main__':
    main()
