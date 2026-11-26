import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BIN_EDGES = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,18,20.])

def prepare_profiles(raw_evolutionary, query_types):
    """Staff preprocessing for the pinned nonnegative ProteinNet score encoding."""
    raw = np.asarray(raw_evolutionary, dtype=np.float64)
    types = np.asarray(query_types)
    if raw.ndim != 2 or raw.shape[1] != 21 or types.shape != (len(raw),):
        raise ValueError("expected (L,21) raw data and L query types")
    if not np.issubdtype(types.dtype, np.integer) or np.any((types < 0) | (types > 20)):
        raise ValueError("query types must be integers from 0 to 20")
    scores = raw[:, :20]  # Deliberately discard the information-content column.
    if np.any(np.isfinite(scores) & (scores < 0)):
        raise ValueError("negative/log-odds encoding needs an explicit conversion")
    valid = np.isfinite(scores).all(-1) & (np.nansum(scores, axis=-1) > 0)
    result = np.eye(21, dtype=np.float32)[types]
    result[valid] = 0
    result[valid, :20] = scores[valid] / scores[valid].sum(-1, keepdims=True)
    return result

def ca_distograms_from_proteinnet(backbone_pm, coord_mask):
    """Staff helper; logical shape (L, atom=N/CA/C, xyz), measured in pm."""
    backbone = np.asarray(backbone_pm, dtype=np.float64)
    mask = np.asarray(coord_mask, dtype=bool)
    if backbone.shape != (len(mask), 3, 3):
        raise ValueError("expected N-CA-C triplets")
    ca = backbone[:, 1, :] / 100.0
    if not np.isfinite(ca[mask]).all():
        raise ValueError("resolved C-alpha coordinates must be finite")
    ca = ca.copy()
    ca[~mask] = 0
    distance = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    labels = np.searchsorted(BIN_EDGES, distance, side="right").astype(np.int64)
    labels[~(mask[:, None] & mask[None, :])] = -1
    return labels

@torch.no_grad()
def canonical_prior(logits):
    """Symmetrize raw/quantized teacher scores and restore log-probability gauge."""
    if logits.ndim != 3 or logits.shape[0] != logits.shape[1] or logits.shape[2] != 16:
        raise ValueError("prior must have shape (L,L,16)")
    if not torch.isfinite(logits).all():
        raise ValueError("prior must be finite")
    logits = logits.detach().float()
    return F.log_softmax(0.5 * (logits + logits.transpose(0, 1)), dim=-1)

def window_loss(logits, labels, valid):
    """r2 CNN/Optimization loss: per-window means over the common central 20x20."""
    if logits.shape[1:] != (16, 32, 32) or labels.shape != logits.shape[:1] + (32, 32):
        raise ValueError("expected Bx16x32x32 logits and Bx32x32 labels")
    spatial = torch.zeros_like(valid, dtype=torch.bool)
    spatial[:, 6:26, 6:26] = True
    valid = valid.bool() & spatial
    losses = []
    for scores, target, selected in zip(logits.permute(0, 2, 3, 1), labels, valid):
        if not selected.any():
            raise ValueError("sampler must supply a nonempty central mask for each window")
        losses.append(F.cross_entropy(scores[selected], target[selected]))
    return torch.stack(losses).mean()

def pair_features(a, residue_index, projection):
    """a is the staff-prepared probability-like descriptor, NOT raw 21D PSSM+IC."""
    s = a @ projection
    gap = (residue_index[:, None] - residue_index[None, :]).abs().to(a.dtype)
    entropy = -torch.special.xlogy(a, a).sum(-1)
    r = torch.stack((gap.clamp_max(32) / 32, gap.log1p() / np.log(97)), -1)
    c = torch.stack((a @ a.T, (entropy[:, None] + entropy[None, :]) / (2 * np.log(21))), -1)
    return torch.cat((s[:, None] + s[None, :], (s[:, None] - s[None, :]).abs(), r, c), -1)

def edge_features(residue_index, prior_logits, edge_index):
    src, dst = edge_index
    q = prior_logits[src, dst].softmax(-1)
    gap = (residue_index[src] - residue_index[dst]).abs().to(q.dtype)
    return torch.stack((q[:, :5].sum(-1), -torch.special.xlogy(q, q).sum(-1) / np.log(16),
                        gap.clamp_max(32) / 32, (gap == 1).to(q.dtype)), -1)

def chain_metrics(logits, labels, coord_mask, residue_index):
    """Single-chain results; dataset-level code averages eligible chains equally."""
    n = len(residue_index)
    gap = (residue_index[:, None] - residue_index[None, :]).abs()
    upper = torch.ones((n, n), dtype=torch.bool, device=logits.device).triu(1)
    resolved = coord_mask[:, None].bool() & coord_mask[None, :].bool()
    valid = upper & resolved & (gap >= 6)
    result = {"nll": None, "precision_at_l": None, "valid_pairs": int(valid.sum())}
    if valid.any():
        # Select first: invalid sentinel labels never reach cross_entropy.
        result["nll"] = float(F.cross_entropy(logits[valid], labels[valid]).detach())
    long_valid = upper & resolved & (gap >= 24)
    candidates = long_valid.nonzero().detach().cpu().tolist()
    score = logits.softmax(-1)[..., :5].sum(-1).detach().cpu()
    idx = residue_index.detach().cpu().tolist()
    candidates.sort(key=lambda ij: (-float(score[ij[0], ij[1]]),
                                    min(idx[ij[0]], idx[ij[1]]), max(idx[ij[0]], idx[ij[1]])))
    selected = candidates[:min(n, len(candidates))]
    if selected:
        result["precision_at_l"] = sum(int(labels[i, j] < 5) for i, j in selected) / len(selected)
    result["long_range_candidates"] = len(candidates)
    return result

