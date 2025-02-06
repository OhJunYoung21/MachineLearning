import os
import torch
import pandas as pd
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nilearn import datasets
from nilearn import plotting
from nilearn import input_data
from nilearn import image
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx

import networkx as nx
## from_numpy_matrix는 더이상 지원되지 않고, 대신 from_numpy_array가 유사한 기능을 수행함.
from networkx.convert_matrix import from_numpy_array

from GNN import DevDataset

dataset = DevDataset('dataset_pyg')
dataset = dataset.shuffle()

train_share = int(len(dataset) * 0.8)

train_dataset = dataset[:train_share]
test_dataset = dataset[train_share:]


class GraphNetwork(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init()

        self.mlp1 = Sequential(Linear(2 * dataset.num_node_features, hidden_channels), ReLU())
        self.mlp2 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())
        self.mlp3 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())
