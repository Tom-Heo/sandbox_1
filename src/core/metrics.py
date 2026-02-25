import torch

class TrimapIoUMetric:
    """[Oxford Pet Trimap 전용 평가지표 누적기]
    에폭(Epoch) 단위로 정확한 IoU를 계산하기 위해 픽셀 단위의 혼동 행렬을 누적합니다.
    """
    def __init__(self, num_classes=3, device='cuda'):
        self.num_classes = num_classes
        self.device = device
        self.reset()

    def reset(self):
        """매 에폭 시작 시 혼동 행렬을 초기화합니다."""
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), device=self.device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        [Fast Histogram Update]
        배치마다 예측값과 정답지의 픽셀 분포를 혼동 행렬에 더합니다.
        
        preds: (B, C, H, W) 형태의 로짓(Logits) 또는 (B, H, W) 형태의 클래스 인덱스
        targets: (B, H, W) 형태의 정답 인덱스
        """
        # 로짓이 들어왔다면 가장 높은 확률의 클래스를 선택 (B, C, H, W) -> (B, H, W)
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)

        # 1차원 텐서로 쭉 펼침 (Flatten)
        preds = preds.flatten()
        targets = targets.flatten()

        # 유효한 정답 클래스만 필터링 (안전장치)
        valid_mask = (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid_mask]
        targets = targets[valid_mask]

        # 핵심 알고리즘: Bincount를 이용한 초고속 혼동 행렬 계산
        # targets * num_classes + preds 공식을 쓰면 2D 행렬의 인덱스를 1D로 맵핑할 수 있습니다.
        indices = self.num_classes * targets + preds
        hist = torch.bincount(indices, minlength=self.num_classes ** 2)
        
        # (num_classes, num_classes) 모양으로 복구 후 누적
        self.confusion_matrix += hist.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """[최종 IoU 계산]
        누적된 혼동 행렬을 바탕으로 클래스별 IoU와 평균 IoU를 산출합니다.
        """
        hist = self.confusion_matrix
        
        # 대각선 성분 = 교집합 (Intersection : 모델과 정답이 일치한 픽셀 수)
        intersection = torch.diag(hist)
        
        # 행의 합 + 열의 합 - 교집합 = 합집합 (Union)
        # hist.sum(dim=1): 정답지(Target)의 픽셀 수
        # hist.sum(dim=0): 모델 예측(Prediction)의 픽셀 수
        union = hist.sum(dim=1) + hist.sum(dim=0) - intersection

        # 0으로 나누는 것을 방지하기 위한 작은 값(eps) 추가
        iou = intersection / (union + 1e-10)

        # 텐서를 파이썬 float으로 변환하여 반환
        metrics = {
            'iou_pet': iou[0].item(),           # 전경 (Class 0)
            'iou_bg': iou[1].item(),            # 배경 (Class 1)
            'iou_boundary': iou[2].item(),      # 경계선 (Class 2) - ★ 이 연구의 핵심 타겟 ★
            'miou': iou.mean().item()           # 전체 평균
        }
        
        return metrics