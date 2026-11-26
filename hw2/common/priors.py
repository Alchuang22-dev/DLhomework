"""Provided bridge from the selected Part Two CNN to frozen Part Three inputs."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from .data import DATA
from .kernels import pair_features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_priors(checkpoint, output, device='cpu', source='part2_plain'):
    """Export all pairs using only node features and original sequence indices.

    A matching training-result JSON is required to identify the plain CNN and
    the validation-selected checkpoint. No target labels or masks are loaded.
    """
    checkpoint, output = Path(checkpoint), Path(output)
    record = json.loads(checkpoint.with_suffix('.json').read_text())
    manifest = json.loads((DATA/'manifest.json').read_text())
    if record['configuration'] != 'plain':
        raise ValueError('Part Three requires the selected Part Two plain CNN')
    if record['metadata']['manifest_sha256'] != sha(DATA/'manifest.json'):
        raise ValueError('CNN checkpoint belongs to a different data manifest')
    if record['checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('CNN checkpoint changed after its training record was saved')
    from CNN.model import DistanceCNN
    model = DistanceCNN('plain').to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval().requires_grad_(False)
    projection = torch.from_numpy(np.load(DATA/'assets/projection.npy')).to(device)
    output.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter(); files = {}
    with torch.no_grad():
        for entry in manifest['entries']:
            with np.load(DATA/entry['file'], allow_pickle=False) as data:
                a = torch.from_numpy(data['node_features'].copy()).to(device)
                idx = torch.from_numpy(data['residue_index'].copy()).to(device)
            features = pair_features(a, idx, projection).permute(2,0,1).contiguous()[None]
            prior = model(features)[0].permute(1,2,0).log_softmax(-1).cpu().numpy()
            prior = prior.astype(np.float32)
            if not np.isfinite(prior).all() or not np.allclose(prior,prior.transpose(1,0,2),atol=1e-6,rtol=0):
                raise ValueError('CNN must produce finite, symmetric full-chain logits')
            file = output/(entry['id']+'.npy'); np.save(file,prior)
            files[entry['id']] = dict(file=file.name,split=entry['split'],
                                      shape=list(prior.shape),sha256=sha(file))
    info = dict(format='hw2-cnn-priors-v1',source=source,cnn_configuration='plain',
                dataset_id=manifest['dataset_id'],data_manifest_sha256=sha(DATA/'manifest.json'),
                cnn_checkpoint_sha256=sha(checkpoint),cnn_updates=record['updates'],
                cnn_selected_step=record['selected_step'],selection_metric=record['selection_metric'],
                cnn_metrics={s:record[s] for s in ['validation','test']},
                storage='float32 log_softmax, all residue pairs',files=files,
                export_seconds=time.perf_counter()-tick)
    (output/'cache.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def validate_prior_cache(directory):
    directory = Path(directory)
    if not (directory/'cache.json').is_file():
        raise FileNotFoundError('CNN prediction cache missing. Run Part Two CNN/train.py --mode all, '
                                'or explicitly use GNN/train.py --use-reference-prior.')
    info=json.loads((directory/'cache.json').read_text())
    manifest=json.loads((DATA/'manifest.json').read_text())
    if info.get('format')!='hw2-cnn-priors-v1' or info.get('cnn_configuration')!='plain':
        raise ValueError('Expected a Part Two plain CNN prediction cache')
    if info['dataset_id']!=manifest['dataset_id'] or info['data_manifest_sha256']!=sha(DATA/'manifest.json'):
        raise ValueError('CNN prediction cache and course data manifest do not match')
    if info['selection_metric']!='full_chain_macro_nll' or info['cnn_updates']!=300:
        raise ValueError('Use the prescribed 300-update, validation-selected CNN')
    if 'cnn_metrics' not in info and info.get('source')!='reference_part2_plain':
        raise ValueError('A student CNN export must include its source metrics')
    if set(info['files'])!={e['id'] for e in manifest['entries']}:
        raise ValueError('CNN prediction cache must cover all 280 chains')
    for entry in manifest['entries']:
        rec=info['files'][entry['id']];name=entry['id']+'.npy'
        if rec['file']!=name or rec['split']!=entry['split'] or rec['shape']!=[entry['length'],entry['length'],16]:
            raise ValueError('Invalid prediction-cache entry: '+entry['id'])
        if sha(directory/name)!=rec['sha256']:
            raise ValueError('Modified CNN prediction: '+name)
    result={k:v for k,v in info.items() if k!='files'}
    result['verified_chains']=len(manifest['entries'])
    return result
