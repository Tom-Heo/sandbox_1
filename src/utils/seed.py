import random
import numpy as np
import torch

def set_seed(seed=42):
    """실험의 완벽한 재현성을 보장하는 시드 고정 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN 결정론적 연산 강제 (속도는 미세하게 저하되나 연산 결과가 100% 동일해짐)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False