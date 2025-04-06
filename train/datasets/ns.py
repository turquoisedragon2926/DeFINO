import os
import torch
import numpy as np
import h5py
from typing import Optional, Dict, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        nx,
        ny,
        eigen_count,
        sample_paths: List[str]):
        self.nx = nx
        self.ny = ny
        self.eigen_count = eigen_count
        self.sample_paths = sample_paths
        
    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        with h5py.File(self.sample_paths[idx], 'r') as f:
            x = f['x'][:]
            y = f['y'][:]
            v = f['v'][:]
            Jvp = f['Jvp'][:]
            
        item = {
            'x': x,
            'y': y,
            'Jvp': Jvp,
            'v': v,
            'sample_path': self.sample_paths[idx]
        }
        return item

class NSDataLoader:
    def __init__(
        self,
        nx: int,
        ny: int,
        eigen_count: int,
        sample_directories: List[str],
        batch_size: int = 1,
        num_workers: int = 1,
        pin_memory: bool = True
    ):
        self.nx = nx
        self.ny = ny
        self.eigen_count = eigen_count
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sample_paths = []
        for directory in sample_directories:
            for file in os.listdir(os.path.join(directory, 'samples')):
                if file.endswith('.h5'):
                    self.sample_paths.append(os.path.join(directory, 'samples', file))
        
    def get_dataloader(self, offset: int, limit: int, shuffle: bool = True):
        dataset = CustomDataset(self.nx, self.ny, self.eigen_count, self.sample_paths[offset:offset+limit])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)        
