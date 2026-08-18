from classes.zarr_class import Zarr
import numpy as np
import torch
import time
import other_scripts.utils as ut
from paths import NC
import random

class ZarrDataset:
        
    def __init__(self, zarr_cube: Zarr, samples):
        self.zarr_cube = zarr_cube
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx): 
        """
        Returns data in sample with index idx
        """
        
        sample = self.samples[idx]
        X = sample["builder"](self.zarr_cube,sample)

        y = sample["label"]
        
        return (X, y)
        

class ZarrDatasetFlip:
        
    def __init__(self, zarr_cube: Zarr, samples):
        self.zarr_cube = zarr_cube
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx): 
        """
        Returns data in sample with index idx
        """
        
        sample = self.samples[idx]
        X = sample["builder"](self.zarr_cube,sample)

        t=random.randint(0,7)
        X = self.apply_transform(X,t)
        y = sample["label"]
        
        return (X, y)
        
    
    def apply_transform(self, X, t):
        # rotate
        k = t % 4
        X = np.rot90(X, k=k, axes=(-2, -1))

        # flip (if t >= 4)
        if t >= 4:
            X = np.flip(X, axis=-1)

        return np.ascontiguousarray(X)

