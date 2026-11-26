"""Student scaffold: implement Adam.step; SGD and Momentum are provided."""
import torch

class BaseOptimizer:

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = float(lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def check_gradients(self):
        if any((p.grad is None for p in self.params)):
            raise ValueError('Every assessed parameter must have a gradient tensor')

class SGD(BaseOptimizer):

    @torch.no_grad()
    def step(self):
        self.check_gradients()
        for p in self.params:
            p.add_(p.grad, alpha=-self.lr)

class Momentum(BaseOptimizer):

    def __init__(self, params, lr, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self.velocity = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        self.check_gradients()
        for p, v in zip(self.params, self.velocity):
            v.mul_(self.momentum).add_(p.grad)
            p.add_(v, alpha=-self.lr)

class Adam(BaseOptimizer):

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params, lr)
        self.betas = tuple(betas)
        self.eps = float(eps)
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.u = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        """TODO Optimization: implement Adam.step according to the handout"""
        raise NotImplementedError('TODO Optimization: implement Adam.step according to the handout')

def make_optimizer(params, name, lr=None):
    if name == 'sgd':
        return SGD(params, 0.05 if lr is None else lr)
    if name == 'momentum':
        return Momentum(params, 0.01 if lr is None else lr)
    if name == 'adam':
        return Adam(params, 0.001 if lr is None else lr)
    raise ValueError(name)
