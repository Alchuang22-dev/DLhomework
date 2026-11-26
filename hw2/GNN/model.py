import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

@torch.no_grad()
def build_edges(residue_index, prior_logits, mode='predicted', k=8):
    """TODO GNN: implement build_edges according to the handout"""
    raise NotImplementedError('TODO GNN: implement build_edges according to the handout')

class MessagePassingLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(68, 32), nn.ReLU(), nn.Linear(32, 32))
        self.update = nn.Linear(64, 32)

    def forward(self, h, edge_index, e):
        """TODO GNN: implement MessagePassingLayer.forward according to the handout"""
        raise NotImplementedError('TODO GNN: implement MessagePassingLayer.forward according to the handout')

class PairHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(80, 32), nn.ReLU(), nn.Linear(32, 16))

    def forward(self, hi, hj, prior):
        """TODO GNN: implement PairHead.forward according to the handout"""
        raise NotImplementedError('TODO GNN: implement PairHead.forward according to the handout')

class DistanceGNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.node = nn.Linear(21, 32)
        self.layers = nn.ModuleList([MessagePassingLayer(), MessagePassingLayer()])
        self.head = PairHead()

    def encode(self, a, edge_index, e):
        h = F.relu(self.node(a))
        for layer in self.layers:
            h = layer(h, edge_index, e)
        return h

    def decode(self, h, pairs, prior_for_pairs):
        i, j = pairs.T
        return self.head(h[i], h[j], prior_for_pairs)

    def forward(self, a, edge_index, e, pairs, prior_for_pairs):
        return self.decode(self.encode(a, edge_index, e), pairs, prior_for_pairs)
