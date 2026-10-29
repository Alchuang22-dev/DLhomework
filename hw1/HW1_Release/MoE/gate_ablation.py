"""Task 3.4.3: change last-layer Top-2 weights while preserving expert IDs.

Run from the release directory:
    python MoE/gate_ablation.py --device cpu

Uses the saved validation-selected seed-416 checkpoint and the official full-test
evaluator. No training, cross-sequence mixing, or checkpoint modification occurs.
"""
import argparse
import copy
import hashlib
import json
import platform
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from train import Corpus, Transformer, evaluate_full, matrix_macs


CONDITIONS = {'original': 1.0, 'uniform': 0.0, 'midpoint': 0.5, 'swapped': -1.0}


def change_weights(gates, condition):
    """Keep expert-slot order; only replace the weights attached to those slots."""
    if condition == 'original':
        return gates
    if condition == 'uniform':
        return torch.full_like(gates, 0.5)
    if condition == 'midpoint':
        return 0.5 * gates + 0.25
    if condition == 'swapped':
        return gates.flip(-1)
    raise ValueError(f'Unknown condition: {condition}')


class RouteAudit:
    """Hash every token's ordered expert IDs and input; collect baseline gaps."""
    def __init__(self, collect_gaps=False):
        self.ids_hash = hashlib.sha256()
        self.inputs_hash = hashlib.sha256()
        self.collect_gaps = collect_gaps
        self.gaps = []
        self.tokens = 0
        self.calls = 0
        self.max_sum_error = 0.0

    def record(self, flat, ids, original_gates, gates):
        self.inputs_hash.update(flat.detach().cpu().contiguous().numpy().tobytes())
        self.ids_hash.update(ids.detach().cpu().contiguous().numpy().tobytes())
        self.tokens += len(flat)
        self.calls += 1
        self.max_sum_error = max(self.max_sum_error,
                                 float((gates.sum(-1) - 1).abs().max()))
        assert bool((gates >= 0).all())
        if self.collect_gaps:
            self.gaps.append((original_gates[:, 0] - original_gates[:, 1]).abs().cpu().numpy())

    def summary(self):
        assert self.max_sum_error < 1e-6
        return {'ordered_expert_ids_sha256': self.ids_hash.hexdigest(),
                'last_ffn_inputs_sha256': self.inputs_hash.hexdigest(),
                'evaluated_tokens': self.tokens, 'route_calls': self.calls,
                'max_gate_sum_error': self.max_sum_error}


@contextmanager
def intervene(layer, condition, audit=None):
    original_route = layer.route

    def route(flat):
        probs, gates, ids = original_route(flat)
        changed = change_weights(gates, condition)
        if audit is not None:
            audit.record(flat, ids, gates, changed)
        return probs, changed, ids

    layer.route = route
    try:
        yield
    finally:
        layer.route = original_route


@torch.no_grad()
def verify(model):
    """Check transformations, weighted outputs, prefix causality, and row independence."""
    gates = torch.tensor([[.8, .2], [.5, .5], [.99, .01]], dtype=torch.float64)
    expected = {
        'original': gates,
        'uniform': torch.full_like(gates, .5),
        'midpoint': torch.tensor([[.65, .35], [.5, .5], [.745, .255]], dtype=torch.float64),
        'swapped': torch.tensor([[.2, .8], [.5, .5], [.01, .99]], dtype=torch.float64),
    }
    for condition in CONDITIONS:
        torch.testing.assert_close(change_weights(gates, condition), expected[condition],
                                   rtol=0, atol=1e-15)

    probe = copy.deepcopy(model).cpu().double().eval()
    layer = probe.blocks[-1].ffn
    assert layer.topk == 2
    torch.manual_seed(73)
    x = torch.randn(7, layer.d, dtype=torch.float64)
    _, original_gates, ids = layer.route(x)
    all_outputs = torch.stack([expert(x) for expert in layer.experts], dim=1)
    length = probe.position.num_embeddings
    inputs = torch.randint(0, 256, (4, length))
    changed = inputs.clone()
    changed[:, length // 2:] = (changed[:, length // 2:] + 127) % 256
    errors = {}
    for condition, lam in CONDITIONS.items():
        # Independent affine definition is the oracle for the implemented cases.
        expected_gates = .5 + lam * (original_gates - .5)
        dense_gates = torch.zeros(7, len(layer.experts), dtype=x.dtype).scatter(1, ids, expected_gates)
        oracle = (all_outputs * dense_gates[:, :, None]).sum(1)
        executed = []
        hooks = [expert.register_forward_pre_hook(lambda module, args: executed.append(len(args[0])))
                 for expert in layer.experts]
        with intervene(layer, condition):
            actual, _ = layer(x)
        for hook in hooks:
            hook.remove()
        assert sum(executed) == len(x) * 2
        torch.testing.assert_close(actual, oracle, rtol=1e-10, atol=1e-12)
        with intervene(layer, condition):
            normal, _ = probe(inputs)
            suffix_changed, _ = probe(changed)
            alone, _ = probe(inputs[:1])
        prefix_error = float((normal[:, :length // 2] - suffix_changed[:, :length // 2]).abs().max())
        row_error = float((normal[:1] - alone).abs().max())
        assert prefix_error < 1e-10 and row_error < 1e-10
        errors[condition] = {'max_output_error': float((actual - oracle).abs().max()),
                             'executed_token_expert_pairs': sum(executed),
                             'max_prefix_logit_error': prefix_error,
                             'max_batch_row_logit_error': row_error}
    torch.testing.assert_close(change_weights(gates, 'swapped').sort(-1).values,
                               gates.sort(-1).values, rtol=0, atol=0)
    return {'passed': True, 'dtype': 'float64 CPU', 'conditions': errors,
            'swapped_weight_multiset_preserved': True}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--root', type=Path, default=root / 'data')
    parser.add_argument('--results', type=Path, default=root / 'results')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint_path = args.results / 'moe_seed416.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    metadata = checkpoint['args']
    domains = metadata['domains'].split(',')
    model = Transformer(metadata['context'], metadata['d'], metadata['width'], True,
                        experts=metadata['experts']).to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
    assert model.blocks[-1].ffn.topk == 2
    verification = verify(model)
    print(json.dumps({'verification': verification}), flush=True)
    corpus = Corpus(args.root / 'corpus.npz', metadata['context'], 'transformer', domains, args.device)
    records = []
    baseline = None
    gaps = None
    for condition, lam in CONDITIONS.items():
        audit = RouteAudit(collect_gaps=condition == 'original')
        with intervene(model.blocks[-1].ffn, condition, audit):
            metrics = evaluate_full(model, corpus, 'test', metadata['batch'])
        record = {'condition': condition, 'lambda': lam, 'test': metrics, 'audit': audit.summary()}
        if condition == 'original':
            baseline = record
            gaps = np.concatenate(audit.gaps)
        else:
            for key in ['ordered_expert_ids_sha256', 'last_ffn_inputs_sha256',
                        'evaluated_tokens', 'route_calls']:
                assert record['audit'][key] == baseline['audit'][key], key
            assert metrics['route_fraction'] == baseline['test']['route_fraction']
            assert metrics['evaluated_targets_per_domain'] == baseline['test']['evaluated_targets_per_domain']
        record['delta_ce'] = metrics['ce'] - baseline['test']['ce']
        record['domain_delta_ce'] = [a - b for a, b in zip(metrics['domain_ce'], baseline['test']['domain_ce'])]
        records.append(record)
        print(json.dumps({k: record[k] for k in ['condition', 'lambda', 'delta_ce', 'domain_delta_ce']} |
                         {'ce': metrics['ce'], 'domain_ce': metrics['domain_ce']}), flush=True)

    previous = json.loads((args.results / 'language.json').read_text())
    old_test = next(r['test'] for r in previous['runs'] if r['mode'] == 'moe' and r['seed'] == 416)
    baseline_error = abs(old_test['ce'] - baseline['test']['ce'])
    assert baseline_error < 1e-6, 'Original condition must reproduce the existing MoE result.'
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), checkpoint['model'][key]), key

    gap_statistics = []
    offset = 0
    edges = np.linspace(0, 1, 21)
    for domain, count in zip(domains, baseline['test']['evaluated_targets_per_domain']):
        values = gaps[offset:offset + count]
        counts, _ = np.histogram(values, bins=edges)
        gap_statistics.append({'domain': domain, 'tokens': count,
                               'mean_gap': float(values.mean(dtype=np.float64)),
                               'median_gap': float(np.median(values)),
                               'p90_gap': float(np.quantile(values, .9)),
                               'fraction_gap_below_0_1': float(np.mean(values < .1)),
                               'histogram_counts': counts.tolist()})
        offset += count
    assert offset == len(gaps)
    result = {
        'protocol': {'model_seed': 416, 'split': 'test', 'domains': domains,
                     'device': args.device, 'machine': platform.machine(),
                     'torch': torch.__version__, 'python': platform.python_version(),
                     'threads': 4, 'batch': metadata['batch'], 'context': metadata['context'],
                     'intervention': 'last-layer gates only; ordered expert IDs unchanged',
                     'formula': 'g_lambda = 0.5 + lambda * (g - 0.5)',
                     'checkpoint_rule': 'existing lowest-validation-CE checkpoint; no retraining',
                     'conditions_specified_before_evaluation': list(CONDITIONS),
                     'matrix_macs_per_token': matrix_macs(metadata['d'], metadata['context'],
                                                        metadata['width'], True, metadata['experts']),
                     'parameters': sum(p.numel() for p in model.parameters()),
                     'interpretation': 'single-checkpoint inference intervention; no training-benefit claim'},
        'verification': verification | {'baseline_ce_absolute_error': baseline_error,
                                        'all_ordered_routes_and_inputs_identical': True,
                                        'all_expert_counts_identical': True,
                                        'checkpoint_parameters_unchanged': True},
        'source_sha256': {'checkpoint': sha256(checkpoint_path),
                          'corpus': sha256(args.root / 'corpus.npz'),
                          **{name: sha256(Path(__file__).parent / name)
                             for name in ['gate_ablation.py', 'model.py', 'train.py']}},
        'conditions': records, 'gap_histogram_edges': edges.tolist(),
        'original_gate_gap_statistics': gap_statistics,
    }
    (args.results / 'gate_ablation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'gate_gap_statistics': gap_statistics}), flush=True)


if __name__ == '__main__':
    main()
