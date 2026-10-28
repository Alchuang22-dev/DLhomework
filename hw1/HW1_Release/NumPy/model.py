"""Expert-module forward and backward propagation for Part Two.

Forward and backward use NumPy without automatic differentiation.

x: (B,D), y: integer labels (B,), k: number of selected experts.
Parameter shapes follow the column-vector convention:
W1 (E,H,D), b1 (E,H), W2 (E,D,H), b2 (E,D),
G (E,D), C (K,D), bc (K,).
Gradient checks use float64 arithmetic.
"""
import numpy as np

def softmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)

def selection_mask(scores, k):
    """Select the k highest-scoring experts in each row.

    Return a Boolean mask with the same shape as scores. Treat the selected
    set as fixed when differentiating within a local selection region.
    """
    ids = np.argsort(-scores, axis=-1, kind='stable')[:, :k]
    mask = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(mask, ids, True, axis=-1)
    return mask

def forward(x, y, params, k, mask=None):
    """Return (mean cross-entropy, cache).

    Use mask when it is not None; otherwise compute it with selection_mask.
    Store 'probs' (B,K) in the cache dictionary, together with any intermediate
    values needed for the backward pass.

    k must be an integer in [1, E]. A supplied mask must be Boolean with
    shape (B,E) and exactly k selected experts per sample.
    """
    batch_size = x.shape[0]
    num_experts = params['G'].shape[0]
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= num_experts:
        raise ValueError('k must be an integer between 1 and the number of experts.')
    if mask is not None:
        if mask.shape != (batch_size, num_experts) or mask.dtype != np.bool_:
            raise ValueError('mask must be a Boolean array with shape (B,E).')
        if not np.all(mask.sum(axis=-1) == k):
            raise ValueError('Each mask row must select exactly k experts.')

    # Each expert maps (B,D) -> (B,H) -> (B,D); e indexes the experts.
    preactivations = np.einsum('bd,ehd->beh', x, params['W1']) + params['b1']
    hidden = np.maximum(preactivations, 0)
    expert_outputs = np.einsum('beh,edh->bed', hidden, params['W2']) + params['b2']

    router_logits = x @ params['G'].T  # (B,E)
    if mask is None:
        mask = selection_mask(router_logits, k)
    # Normalize over selected experts only; all other gate weights are zero.
    gates = softmax(np.where(mask, router_logits, -np.inf))
    residual = x + np.einsum('be,bed->bd', gates, expert_outputs)  # (B,D)

    logits = residual @ params['C'].T + params['bc']
    # Compute log-probabilities directly to avoid taking log of an underflowed probability.
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    loss = -log_probs[np.arange(batch_size), y].mean()
    cache = dict(
        v=preactivations,
        u=hidden,
        experts=expert_outputs,
        gates=gates,
        residual=residual,
        probs=np.exp(log_probs),
    )
    return loss, cache


def backward(x, y, params, cache):
    """Return (parameter_grads, input_grad).

    parameter_grads has the same keys/shapes as params; input_grad is (B,D).
    Include expert, router and residual contributions to input_grad.
    """
    batch_size = x.shape[0]
    # Mean cross-entropy: divide by B once, then propagate this scaling.
    d_logits = cache['probs'].copy()  # Preserve cached predictions.
    d_logits[np.arange(batch_size), y] -= 1
    d_logits /= batch_size
    grads = {
        'C': d_logits.T @ cache['residual'],
        'bc': d_logits.sum(axis=0),
    }
    d_residual = d_logits @ params['C']  # (B,D)

    # Expert path: weighted outputs -> second FC -> ReLU -> first FC.
    gates = cache['gates']
    d_expert_outputs = gates[:, :, None] * d_residual[:, None, :]  # (B,E,D)
    grads['W2'] = np.einsum('bed,beh->edh', d_expert_outputs, cache['u'])
    grads['b2'] = d_expert_outputs.sum(axis=0)
    d_hidden = np.einsum('bed,edh->beh', d_expert_outputs, params['W2'])
    d_preactivations = d_hidden * (cache['v'] > 0)  # ReLU'(0) = 0.
    grads['W1'] = np.einsum('beh,bd->ehd', d_preactivations, x)
    grads['b1'] = d_preactivations.sum(axis=0)

    # Router path: differentiate the selected softmax with the mask fixed.
    d_gates = np.einsum('bd,bed->be', d_residual, cache['experts'])
    d_router_logits = gates * (d_gates - (gates * d_gates).sum(axis=-1, keepdims=True))
    grads['G'] = d_router_logits.T @ x

    # The shared input receives three separate contributions, each (B,D).
    d_input_experts = np.einsum('beh,ehd->bd', d_preactivations, params['W1'])
    d_input_router = d_router_logits @ params['G']
    d_input_residual = d_residual
    d_input = d_input_experts + d_input_router + d_input_residual
    return grads, d_input
