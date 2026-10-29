params = {
    "kernel": kernel,
    "expert_w1": expert_w1,  # Includes all experts; likewise for other expert parameters.
    "expert_b1": expert_b1,
    "expert_w2": expert_w2,
    "expert_b2": expert_b2,
    "router": router,
    "head": head,
    "head_bias": head_bias,
}
for batch in training_batches:
    prob, cache = forward_all(batch.inputs, params)
    loss = mean_cross_entropy(prob, batch.targets)
    logits_grad = (prob - batch.one_hot_targets) / batch.size
    grads, encoded_grad = backward_head_experts_router_res(logits_grad, cache, params)
    conv_grad = encoded_grad * (cache.conv_pre > 0)  # (m, 2)
    # (m, 2, 1) * (m, 2, 3): sum over samples and windows, keeping the kernel axis.
    # The upstream gradient already includes 1/m; do not average again.
    grads["kernel"] = (conv_grad[..., None] * cache.input_windows).sum(axis=(0, 1))
    for name in params:
        params[name] -= learning_rate * grads[name]

