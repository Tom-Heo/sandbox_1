import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    [경계선 타겟팅 무기 1: Focal Loss]
    예측 확률이 낮고 헷갈리는 픽셀(Hard Example)에 페널티를 기하급수적으로 먹입니다.
    Trimap 데이터셋에서 모델이 가장 헷갈려하는 부분은 100% '경계선'입니다.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # alpha: 클래스별 가중치 (예: 경계선 클래스에 더 큰 가중치 부여)
        self.register_buffer('alpha', alpha)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # inputs: (B, C, H, W) - 로짓(Logits) 상태
        # targets: (B, H, W) - 정수형 클래스 인덱스 {0, 1, 2}
        
        # 1. 기본 Cross Entropy Loss 계산 (픽셀 단위)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # 2. 정답 클래스에 대한 예측 확률(pt) 추출
        pt = torch.exp(-ce_loss)

        # 3. Focal Loss 수식 적용: (1 - pt)^gamma * CE
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    [경계선 타겟팅 무기 2: Multiclass Dice Loss]
    전경과 배경의 '덩어리(Volume)'를 빠르고 안정적으로 잡아내기 위한 Loss입니다.
    클래스 불균형(전경/배경 픽셀 >>> 경계선 픽셀)에 매우 강합니다.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # 1. 로짓을 확률값으로 변환
        probs = F.softmax(inputs, dim=1)
        
        # 2. 정답지(targets)를 원-핫 인코딩 형태로 변환: (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # 3. 클래스별 Dice 계수 계산
        dims = (0, 2, 3) # Batch, Height, Width 축으로 합산 (채널(클래스)별 독립 계산)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # 4. Loss로 변환 (1 - Dice) 및 클래스 평균
        return 1.0 - dice_score.mean()


class BoundaryTargetedLoss(nn.Module):
    """
    [최종 결합 손실 함수]
    Focal Loss와 Dice Loss를 1:1로 결합하되, 
    알파(Alpha) 가중치를 통해 '경계선(Boundary, Class 2)'에 대한 페널티를 2배로 증폭시킵니다.
    """
    def __init__(self, focal_weight=0.5, dice_weight=0.5, boundary_boost=2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        # Class 0: Pet(전경), Class 1: Background(배경), Class 2: Boundary(경계선)
        # 경계선 픽셀을 틀렸을 때의 페널티를 boundary_boost 배로 높입니다.
        alpha_weights = torch.tensor([1.0, 1.0, boundary_boost], dtype=torch.float32)
        
        self.focal = FocalLoss(alpha=alpha_weights, gamma=2.0)
        self.dice = DiceLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # 텐서 디바이스 동기화 (가중치 텐서를 입력 텐서와 같은 GPU로 이동)
        if self.focal.alpha is not None and self.focal.alpha.device != inputs.device:
            self.focal.alpha = self.focal.alpha.to(inputs.device)

        l_focal = self.focal(inputs, targets)
        l_dice = self.dice(inputs, targets)
        
        return (self.focal_weight * l_focal) + (self.dice_weight * l_dice)