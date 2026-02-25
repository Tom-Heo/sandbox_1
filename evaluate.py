import os
import torch
from src.data.dataset import get_dataloaders
from src.models.unet import PetSegmentationModel
from src.core.metrics import TrimapIoUMetric

def evaluate_model(weight_path, use_oklab, use_helu, device='cuda'):
    print(f"\n🔍 Evaluating: {os.path.basename(weight_path)}")
    
    # 1. 모델 로드
    model = PetSegmentationModel(use_oklab=use_oklab, use_helu=use_helu).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 2. Test 데이터로더 (Batch=16)
    _, val_loader = get_dataloaders(batch_size=16)
    metric = TrimapIoUMetric(num_classes=3, device=device)

    # 3. 추론 및 혼동 행렬 누적
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            metric.update(outputs, masks)

    # 4. 결과 출력
    results = metric.compute()
    print("-" * 40)
    print(f"✅ Pet (Foreground) IoU : {results['iou_pet']:.4f}")
    print(f"✅ Background IoU       : {results['iou_bg']:.4f}")
    print(f"🔥 Boundary IoU         : {results['iou_boundary']:.4f}  <-- Core Metric")
    print(f"📊 Mean IoU (mIoU)      : {results['miou']:.4f}")
    print("-" * 40)

if __name__ == '__main__':
    # 평가를 원하는 모델의 가중치 경로와 설정을 입력하세요.
    evaluate_model(
        weight_path="outputs/weights/OklabP_HeLU_best.pth",
        use_oklab=True,
        use_helu=True
    )