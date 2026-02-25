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
    """[완벽한 재현성 통제]"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 0. 전역 통제 설정
    set_seed(42)
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 500,
        "batch_size": 16,  # 4개 모델 동시 학습이므로 실제 VRAM은 약 30~40GB 점유 예상
    }

    print(
        f"🚀 Hardware Check: Using {config['device'].upper()} with Concurrent Training"
    )

    # 데이터로더 생성 (단 1개의 배치 스트림)
    train_loader, val_loader = get_dataloaders(batch_size=config["batch_size"])

    # 1. 2x2 요인 설계 실험 목록
    experiments = [
        {"name": "sRGB_ReLU", "use_oklab": False, "use_helu": False},
        {"name": "sRGB_HeLU", "use_oklab": False, "use_helu": True},
        {"name": "OklabP_ReLU", "use_oklab": True, "use_helu": False},
        {"name": "OklabP_HeLU", "use_oklab": True, "use_helu": True},
    ]

    # 2. 4개 모델의 독립적인 객체들을 담을 딕셔너리 준비
    models = {}
    optimizers = {}
    schedulers = {}
    metrics = {}
    visualizers = {}
    histories = {}
    best_ious = {}

    criterion = BoundaryTargetedLoss(boundary_boost=2.0).to(config["device"])

    print("\n📦 Initializing 4 Models into VRAM...")
    for exp in experiments:
        name = exp["name"]
        model = PetSegmentationModel(
            use_oklab=exp["use_oklab"], use_helu=exp["use_helu"]
        ).to(config["device"])

        models[name] = model
        optimizers[name] = build_optimizer(model, base_lr=1e-4)
        schedulers[name] = build_scheduler(optimizers[name], warmup_epochs=5)
        metrics[name] = TrimapIoUMetric(num_classes=3, device=config["device"])
        visualizers[name] = ResearchVisualizer(save_dir=f"outputs/figures/{name}")

        histories[name] = {"train_loss": [], "val_boundary_iou": [], "val_miou": []}
        best_ious[name] = 0.0

    os.makedirs("outputs/weights", exist_ok=True)

    # 3. 오케스트레이션 (동시 학습 루프)
    for epoch in range(1, config["epochs"] + 1):
        print(f"\n{'='*60}\n🏁 Epoch[{epoch}/{config['epochs']}]\n{'='*60}")

        # -------------------[TRAIN PHASE] -------------------
        for name in models:
            models[name].train()

        train_losses = {name: 0.0 for name in models}

        pbar = tqdm(train_loader, desc="[Train]", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(config["device"]), masks.to(config["device"])

            # 단일 배치를 4개 모델이 동시에 먹고 각각 역전파 수행
            for name, model in models.items():
                optimizers[name].zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizers[name].step()

                train_losses[name] += loss.item()

        # Train 에폭 종료 처리
        for name in models:
            avg_train_loss = train_losses[name] / len(train_loader)
            histories[name]["train_loss"].append(avg_train_loss)
            schedulers[name].step()

        # -------------------- [VAL PHASE] --------------------
        for name in models:
            models[name].eval()
            metrics[name].reset()

        with torch.no_grad():
            for i, (imgs, masks) in enumerate(
                tqdm(val_loader, desc="[Valid]", leave=False)
            ):
                imgs, masks = imgs.to(config["device"]), masks.to(config["device"])

                for name, model in models.items():
                    outputs = model(imgs)
                    metrics[name].update(outputs, masks)

                    # 에폭별 첫 번째 배치에서 4개 모델 모두 시각화용 이미지 추출
                    if i == 0:
                        preds = torch.argmax(outputs, dim=1)
                        visualizers[name].save_prediction_grid(
                            epoch,
                            imgs.cpu(),
                            masks.cpu(),
                            preds.cpu(),
                            filename=f"epoch_{epoch:03d}.png",
                        )

        # -----------------[LOGGING & SAVE] -----------------
        print(f"\n📊 Epoch [{epoch}] Summary:")
        for name in models:
            res = metrics[name].compute()
            b_iou = res["iou_boundary"]
            m_iou = res["miou"]
            current_lr = schedulers[name].get_last_lr()[0]

            histories[name]["val_boundary_iou"].append(b_iou)
            histories[name]["val_miou"].append(m_iou)

            # 결과 출력 (터미널에서 4개 모델을 한눈에 비교)
            print(
                f"  [{name:<12}] Loss: {histories[name]['train_loss'][-1]:.4f} | "
                f"LR: {current_lr:.2e} | mIoU: {m_iou:.4f} | Boundary IoU: {b_iou:.4f}"
            )

            # 최고 성능 갱신 시 가중치 저장 (오직 경계선 IoU 기준)
            if b_iou > best_ious[name]:
                best_ious[name] = b_iou
                save_path = f"outputs/weights/{name}_best.pth"
                torch.save(models[name].state_dict(), save_path)
                print(f"      ⭐ {name} updated best weights! (B-IoU: {b_iou:.4f})")

    # 4. 최종 논문용 4색 그래프 렌더링
    print("\n🎨 Rendering Final Convergence Graph for Paper...")
    final_visualizer = ResearchVisualizer(save_dir="outputs/figures")

    # 시각화 함수 요구 포맷으로 히스토리 변환
    plot_data = {
        name: {"boundary_iou": hist["val_boundary_iou"]}
        for name, hist in histories.items()
    }

    final_visualizer.plot_4model_iou_curves(
        plot_data, warmup_epochs=5, filename="Final_Boundary_IoU_Convergence.pdf"
    )
    print("✅ All 4 Experiments Completed Concurrently!")


if __name__ == "__main__":
    main()
