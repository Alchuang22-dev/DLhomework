"""Expert-module forward and backward propagation for Part Two.

Complete forward and backward using NumPy without automatic differentiation.

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
    """
    raise NotImplementedError('Implement expert, gate, residual and head forward passes.')

def backward(x, y, params, cache):
    """Return (parameter_grads, input_grad).

    parameter_grads has the same keys/shapes as params; input_grad is (B,D).
    Include expert, router and residual contributions to input_grad.
    """
    raise NotImplementedError('Implement the gradients derived in Part One.')
