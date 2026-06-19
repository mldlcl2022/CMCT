import os
import torch
import numpy as np

def seed(num):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(num)
    
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only= True)
    
    np.random.seed(num)