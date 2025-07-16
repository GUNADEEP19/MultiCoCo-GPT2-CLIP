import random
import numpy as np
import torch

class Config:
    def __init__(self, d):
        self.__dict__.update(d)
    def __getitem__(self, k):
        return getattr(self, k)
    def __contains__(self, k):
        return hasattr(self, k)
    def __repr__(self):
        return str(self.__dict__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 