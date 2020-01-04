import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import devo.DE as DE
import numpy as np
import ctypes as c
import math


class Net(nn.Sequential):
    def __init__(self, N, D_in, H, D_out, optimizer):
        super(Net, self).__init__()

        # Problem size / parameters
        self.N = N
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.opt = optimizer

        self.fc1 = nn.Linear(D_in, N)
        self.fc2 = nn.Linear(N, H)
        self.fc3 = nn.Linear(H, D_out)

        # init min reward
        self.min_reward = float('inf')
        self.min_weights = None

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(self.fc3(x))
        return x
        # return F.log_softmax(x, dim=1)

    def flatten(self):
        flattened = self._flatten_all()
        self.flattened = flattened
        return flattened

    def tensor_list(self):
        # store tensors / dimensions
        return self.parameters()

    def sizes(self):
        sizes_ = []
        tensors = self.tensor_list()
        for p in self.parameters():
            sizes_.append(p.size())
        return sizes_

    def _flatten_all(self):
        t_list = self.tensor_list()
        sizes = []  # preserve sizes, assuming they're immutable
        flats = []
        for a in t_list:
            size = a.size()
            sizes.append(size)
            a_f = a.flatten()
            flats.append(a_f.detach().numpy())
        flattened = np.concatenate(flats)
        return flattened

    def unflatten_from_flattened_list(self, flattened_t_list):
        t_list = flattened_t_list
        sizes = self.sizes()  # this assumes sizes are immutable
        tensors = []
        offset = 0
        for i in range(0, len(sizes)):
            size = sizes[i]
            total = 1
            for n in size:
                total = total * n
            next_offset = offset + total
            slice_ = t_list[offset:next_offset]
            t_flat = torch.FloatTensor(slice_)
            t_shaped = t_flat.view(size)
            tensors.append(t_shaped)
            offset = next_offset

        # TODO: create systematic way to update the nn from the tensor array
        return tensors

    def update_from_unflattened_tensors(self, unflattened_tensors):
        i = 0
        for p in self.parameters():
            p.data = unflattened_tensors[i]
            i += 1
        return self.parameters()

    # Use numpy vector to update weights

    def update_weights_from_vec(self, weights_vec):
        weights_tensor = torch.from_numpy(weights_vec).float()
        tensor_list = self.unflatten_from_flattened_list(weights_tensor)
        self.update_from_unflattened_tensors(tensor_list)
