import os
import numpy as np

import torch.nn.functional as F
import torch
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx

from GNN.Data_preparation import root_dir
from GNN.Data_preparation import corr_matrices_dir, pcorr_matrices_dir
from GNN.Data_preparation import labels_file
import networkx as nx
from networkx.convert_matrix import from_numpy_array


class DevDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, neighbors=10):
        self.neighbors = neighbors
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):

        """ Converts raw data into GNN-readable format by constructing
                graphs out of connectivity matrices.
        """

        corr_path_list = sorted(os.listdir(corr_matrices_dir), key=lambda x: int(x[5:7]))
        pcorr_path_list = sorted(os.listdir(pcorr_matrices_dir), key=lambda x: int(x[6:8]))

        graph = []

        labels = torch.from_numpy(np.loadtxt(labels_file, delimiter=','))

        for i in range(0, len(corr_path_list)):
            corr_matrix_path = os.path.join(corr_matrices_dir, corr_path_list[i])
            pcorr_matrix_path = os.path.join(pcorr_matrices_dir, pcorr_path_list[i])

            pcorr_matrix_np = np.loadtxt(pcorr_matrix_path, delimiter=',')

            index = np.abs(pcorr_matrix_np).argsort(axis=1)
            n_rois = pcorr_matrix_np.shape[0]

            ### Take only top 10(self.neighbors) correlates to reduce number of edges

            for j in range(n_rois):
                for k in range(n_rois - self.neighbors):
                    pcorr_matrix_np[j, index[j, k]] = 0
                for k in range(n_rois - self.neighbors):
                    pcorr_matrix_np[j, index[j, k]] = 1

            ### Convert numpy array into Graph

            pcorr_matrix_nx = from_numpy_array(pcorr_matrix_np)
            pcorr_matrix_data = from_networkx(pcorr_matrix_nx)

            # Correlation matrix which will serve as our features

            corr_matrix_np = np.loadtxt(corr_matrix_path, delimiter=',')

            pcorr_matrix_data.x = torch.tensor(corr_matrix_np).float()
            pcorr_matrix_data.y = labels[i].type(torch.LongTensor)

            ### Add to running list of all dataset items

            graph.append(pcorr_matrix_data)

        data, slices = self.collate(graph)
        torch.save((data, slices), self.processed_paths[0])


dataset = DevDataset('dataset_pyg')
dataset = dataset.shuffle()

# Train/test split (80-20)
train_share = int(len(dataset) * 0.8)

train_dataset = dataset[:train_share]
test_dataset = dataset[train_share:]
