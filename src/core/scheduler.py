import torch
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR


def build_scheduler(
    optimizer: torch.optim.Optimizer, warmup_epochs: int = 5, gamma: float = 0.95
):
    """
    [Warm-up + Exponential Decay 스케줄러]

    1. Warm-up (0 ~ warmup_epochs):
       초기 학습률의 1%에서 시작하여 목표 학습률(100%)까지 선형적으로 부드럽게 상승시킵니다.
       HeLU2d의 초기 파라미터들이 모델 전체를 망가뜨리지 않고 제자리를 찾게 해줍니다.

    2. Exponential Decay (warmup_epochs ~ ):
       이후 매 에폭마다 이전 에폭 학습률의 `gamma` 비율(예: 95%)만큼 부드럽게 감소시킵니다.
       경계선(Boundary)의 1~2 픽셀 차이를 미세 조정(Fine-tuning)하기 위한 필수 조건입니다.
    """

    # 1. 웜업 스케줄러: 목표 학습률의 0.01배에서 시작하여 1.0배까지 선형 증가
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )

    # 2. 지수 감쇠 스케줄러: 매 에폭마다 gamma(예: 0.95)를 곱하여 감쇠
    decay_scheduler = ExponentialLR(optimizer, gamma=gamma)

    # 3. 두 스케줄러를 순차적으로 결합 (milestones 시점에서 스위치)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_epochs],
    )

    return scheduler


# ---------------------------------------------------------
# [사용 예시 및 Optimizer 설정 가이드]
# ---------------------------------------------------------
def build_optimizer(
    model: torch.nn.Module, base_lr: float = 1e-4, weight_decay: float = 1e-4
):
    """
    HeLU2d 파라미터(알파, 베타, 가중치 등)는 0으로 쪼그라들지 않도록
    Weight Decay를 해제(0.0)하는 그룹 분리형 Optimizer 팩토리입니다.
    """
    helu_params = []
    base_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # HeLU2d 내부 파라미터인지 이름으로 필터링
        if (
            "alpha" in name
            or "beta" in name
            or "redweight" in name
            or "blueweight" in name
        ):
            helu_params.append(param)
        else:
            base_params.append(param)

    # HeLU2d 파라미터는 Weight Decay를 0으로 주고, 학습률을 10배 키워 더 빠르게 변형되도록 유도
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": helu_params, "lr": base_lr * 256.0, "weight_decay": 0.0},
        ]
    )

    return optimizer
