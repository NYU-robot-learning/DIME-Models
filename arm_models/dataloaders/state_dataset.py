import os
import torch
from torch.utils.data import Dataset

from arm_models.utils.load_data import load_state_actions

class CubeRotationDataset(Dataset):
    def __init__(self, data_path = '/home/sridhar/dexterous_arm/models/arm_models/data/cube_rotation/complete'):
        self.data_path = data_path
        self.states, self.actions = load_state_actions(data_path)
    
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.states[idx], self.actions[idx]

class ObjectFlippingDataset(Dataset):
    def __init__(self, data_path = '/home/sridhar/dexterous_arm/models/arm_models/data/object_flipping/complete'):
        self.data_path = data_path
        self.states, self.actions = load_state_actions(data_path)
    
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.states[idx], self.actions[idx]

class FidgetSpinningDataset(Dataset):
    def __init__(self, data_path = '/home/sridhar/dexterous_arm/models/arm_models/data/fidget_spinning/complete'):
        self.data_path = data_path
        self.states, self.actions = load_state_actions(data_path)
    
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.states[idx], self.actions[idx]
