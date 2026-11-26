"""Complete task models; optimizer assessment needs no student CNN/GNN code."""
from pathlib import Path
import torch

def load_model(device='cpu',task='gnn'):
    path=Path(__file__).resolve().parents[1]/f'data/assets/optimization_{task}.pt'
    return torch.jit.load(str(path),map_location=device)

def load_graph_builder(device='cpu'):
    path=Path(__file__).resolve().parents[1]/'data/assets/optimization_graph.pt'
    return torch.jit.load(str(path),map_location=device)
