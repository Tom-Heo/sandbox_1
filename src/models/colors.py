import os
import torch
import random
import numpy as np
from tqdm import tqdm

# 내부 모듈 임포트
from src.data.dataset import get_dataloaders
from src.models.unet import PetSegmentationModel
from src.core.loss import BoundaryTargetedLoss
from src.core.metrics import TrimapIoUMetric
from src.core.scheduler import build_optimizer, build_scheduler
from src.utils.visualizer import ResearchVisualizer


def set_seed(seed=42):
    """[완벽한 재현성 통제]
    운(Luck)이 개입할 여지를 원천 차단합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_model(exp_name, use_oklab, use_helu, dataloaders, config):
    """단일 모델의 학습부터 평가, 시각화, 가중치 저장까지 책임지는 파이프라인"""
    print(f"\n{'='*50}\n🚀 Starting Experiment: {exp_name}\n{'='*50}")

    device = config["device"]
    epochs = config["epochs"]
    train_loader, val_loader = dataloaders

    # 1. 아키텍처, 손실함수, 평가망, 옵티마이저, 스케줄러 세팅
    model = PetSegmentationModel(use_oklab=use_oklab, use_helu=use_helu).to(device)
    criterion = BoundaryTargetedLoss(boundary_boost=2.0).to(device)
    metric = TrimapIoUMetric(num_classes=3, device=device)
    visualizer = ResearchVisualizer(save_dir=f"outputs/figures/{exp_name}")

    optimizer = build_optimizer(model, base_lr=1e-4)
    scheduler = build_scheduler(optimizer, warmup_epochs=5)

    best_boundary_iou = 0.0
    history = {"train_loss": [], "val_boundary_iou": [], "val_miou": []}

    for epoch in range(1, epochs + 1):
        # ------------------- [TRAIN PHASE] -------------------
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()  # 에폭 종료 후 스케줄러 업데이트

        # -------------------- [VAL PHASE] --------------------
        model.eval()
        metric.reset()

        with torch.no_grad():
            for i, (imgs, masks) in enumerate(val_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                metric.update(outputs, masks)

                # 에폭별 첫 번째 배치에서 시각화용 이미지 추출 (추이 관찰용)
                if i == 0:
                    preds = torch.argmax(outputs, dim=1)
                    visualizer.save_prediction_grid(
                        epoch,
                        imgs.cpu(),
                        masks.cpu(),
                        preds.cpu(),
                        filename=f"epoch_{epoch:03d}.png",
                    )

        metrics = metric.compute()
        b_iou = metrics["iou_boundary"]

        history["train_loss"].append(avg_train_loss)
        history["val_boundary_iou"].append(b_iou)
        history["val_miou"].append(metrics["miou"])

        print(
            f"Epoch[{epoch}/{epochs}] "
            f"Loss: {avg_train_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"mIoU: {metrics['miou']:.4f} | "
            f"Boundary IoU: {b_iou:.4f}"
        )

        # ----------------- [EARLY STOPPING & SAVE] -----------------
        # 조기 종료 및 가중치 저장의 기준은 오직 '경계선(Boundary) IoU'입니다.
        if b_iou > best_boundary_iou:
            best_boundary_iou = b_iou
            os.makedirs("outputs/weights", exist_ok=True)
            save_path = f"outputs/weights/{exp_name}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"🌟 Best Model Saved! (Boundary IoU: {best_boundary_iou:.4f})")

    return history


def main():
    # 0. 전역 통제 설정
    set_seed(42)
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 50,  # 연구 목적이므로 충분히 길게 돌립니다.
        "batch_size": 16,  # 기울기 노이즈 보존을 위해 16으로 통제
    }

    print(f"Hardware Check: Using {config['device'].upper()}")

    # 데이터로더는 단 한 번만 생성하여 4개 모델이 완전히 동일한 난수 배치를 먹게 합니다.
    dataloaders = get_dataloaders(batch_size=config["batch_size"])

    # 1. 2x2 요인 설계 (Factorial Design) 실험 목록
    experiments = [
        {"name": "sRGB_ReLU", "use_oklab": False, "use_helu": False},
        {"name": "sRGB_HeLU", "use_oklab": False, "use_helu": True},
        {"name": "OklabP_ReLU", "use_oklab": True, "use_helu": False},
        {"name": "OklabP_HeLU", "use_oklab": True, "use_helu": True},
    ]

    all_histories = {}

    # 2. 오케스트레이션 (순차 학습)
    for exp in experiments:
        history = train_single_model(
            exp_name=exp["name"],
            use_oklab=exp["use_oklab"],
            use_helu=exp["use_helu"],
            dataloaders=dataloaders,
            config=config,
        )
        # 시각화 툴 포맷에 맞게 변환
        all_histories[exp["name"]] = {
            "boundary_loss": history[
                "train_loss"
            ]  # 단순화를 위해 train_loss를 대표로 사용
        }

    # 3. 최종 논문용 4색 그래프 렌더링
    print("\n🎨 Rendering Final Convergence Graph for Paper...")
    final_visualizer = ResearchVisualizer(save_dir="outputs/figures")
    final_visualizer.plot_4model_loss_curves(
        all_histories, warmup_epochs=5, filename="Final_Loss_Convergence.pdf"
    )
    print("✅ All Experiments Completed Successfully!")


if __name__ == "__main__":
    main()
