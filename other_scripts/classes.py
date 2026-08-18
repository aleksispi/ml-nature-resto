import os, sys
from typing import List, Tuple, Optional
import rasterio
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import random
from collections import OrderedDict
import gc

# Simple 5-layer MLP model (used as basis for cloud optical thickness prediction)
class MLP5(nn.Module):
	def __init__(self, input_dim, output_dim=1, hidden_dim=64, apply_relu=True):
		super(MLP5, self).__init__()
		self.lin1 = nn.Linear(input_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim)
		self.lin3 = nn.Linear(hidden_dim, hidden_dim)
		self.lin4 = nn.Linear(hidden_dim, hidden_dim)
		self.lin5 = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()
		self.apply_relu = apply_relu

	def forward(self, x):
		x1 = self.lin1(x)
		x1 = self.relu(x1)
		x2 = self.lin2(x1)
		x2 = self.relu(x2)
		x3 = self.lin3(x2)
		x3 = self.relu(x3)
		x4 = self.lin4(x3)
		x4 = self.relu(x4)
		x5 = self.lin5(x4)
		if self.apply_relu:
			x5[:, 0] = self.relu(x5[:, 0])  # NB: cloud optical thicknesses cannot be negative
		return x5

def replace(string_in, replace_from, replace_to='_'):
    if not isinstance(replace_from, list):
        replace_from = [replace_from]
    string_out = string_in
    for replace_entry in replace_from:
        string_out = string_out.replace(replace_entry, replace_to)
    return string_out