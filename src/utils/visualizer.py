import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class ResearchVisualizer:
    """[논문용 고해상도 시각화 및 에러 맵 생성기]"""
    
    def __init__(self, save_dir='outputs/figures'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Trimap 클래스 색상 지정 (논문 가독성을 위한 배색)
        # 0(Pet): 투명한 초록, 1(Bg): 투명한 파랑, 2(Boundary): 쨍한 노랑
        self.cmap = mcolors.ListedColormap(['#2ca02c', '#1f77b4', '#ff7f0e'])

    def _unnormalize(self, tensor: torch.Tensor):
        """[-1, 1]로 정규화된 텐서를 시각화를 위해[0, 1]로 복구합니다."""
        return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)

    def generate_error_map(self, img_np, gt_np, pred_np):
        """
        [가장 중요한 연구 무기: Error Map]
        원본 이미지를 흑백(Grayscale)으로 깔고, 모델이 틀린 부분만 색칠합니다.
        - 특히 '경계선(Boundary)'을 틀린 픽셀은 강렬한 빨간색(Red)으로 칠해
          HeLU2d+OklabP가 이 붉은 띠를 얼마나 얇게 깎아냈는지 증명합니다.
        """
        # 1. 흑백 배경 만들기 (H, W, 3)
        gray_img = np.dot(img_np[..., :3],[0.2989, 0.5870, 0.1140])
        error_map = np.stack((gray_img,)*3, axis=-1) * 0.5  # 대비를 낮춰 에러가 돋보이게 함

        # 2. 에러 마스크 추출
        wrong_mask = (gt_np != pred_np)
        boundary_gt_mask = (gt_np == 2)
        
        # 일반 클래스(전경/배경) 에러 -> 보라색 
        error_map[wrong_mask] =[0.5, 0.0, 0.5]
        
        # 경계선 클래스 에러 (치명적 에러) -> 강렬한 빨간색
        boundary_error = wrong_mask & boundary_gt_mask
        error_map[boundary_error] = [1.0, 0.0, 0.0]

        return error_map

    def save_prediction_grid(self, epoch, img, gt, pred, filename="pred_grid.png"):
        """
        [1x4 Grid] 원본 | 정답 | 예측 | 에러맵
        """
        # 텐서를 넘파이로 변환 (CPU 이동)
        img_np = self._unnormalize(img[0]).permute(1, 2, 0).cpu().numpy()
        gt_np = gt[0].cpu().numpy()
        pred_np = pred[0].cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
        plt.subplots_adjust(wspace=0.05)

        # 1. Original
        axes[0].imshow(img_np)
        axes[0].set_title("Input (sRGB)", fontsize=12)

        # 2. Ground Truth
        axes[1].imshow(img_np)
        axes[1].imshow(gt_np, cmap=self.cmap, alpha=0.5, interpolation='nearest')
        axes[1].set_title("Ground Truth (Trimap)", fontsize=12)

        # 3. Prediction
        axes[2].imshow(img_np)
        axes[2].imshow(pred_np, cmap=self.cmap, alpha=0.5, interpolation='nearest')
        axes[2].set_title(f"Prediction (Epoch {epoch})", fontsize=12)

        # 4. Error Map
        error_map = self.generate_error_map(img_np, gt_np, pred_np)
        axes[3].imshow(error_map)
        axes[3].set_title("Error Map (Red=Boundary Error)", fontsize=12)

        for ax in axes:
            ax.axis('off')

        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches='tight')
        plt.close()

    def plot_4model_loss_curves(self, histories: dict, warmup_epochs=5, filename="loss_curves.pdf"):
        """
        [논문 본문용 4개 모델 통합 Loss 비교 그래프]
        Y축을 Log-Scale로 변환하고, Warm-up 구간을 시각적으로 하이라이트합니다.
        
        histories = {
            'sRGB_ReLU': {'boundary_loss': [...], ...},
            'OklabP_HeLU': {'boundary_loss': [...], ...}, ...
        }
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # 선 스타일 매핑 (통제 변인에 따른 일관된 디자인)
        styles = {
            'sRGB_ReLU':   {'color': 'gray',  'ls': '--', 'lw': 1.5, 'label': 'Baseline (sRGB + ReLU)'},
            'sRGB_HeLU':   {'color': 'blue',  'ls': '-',  'lw': 2.0, 'label': 'sRGB + HeLU2d'},
            'OklabP_ReLU': {'color': 'green', 'ls': '-',  'lw': 2.0, 'label': 'OklabP + ReLU'},
            'OklabP_HeLU': {'color': 'red',   'ls': '-',  'lw': 2.5, 'label': 'Proposed (OklabP + HeLU2d)'}
        }

        # Warm-up 구간 음영 처리
        ax.axvspan(0, warmup_epochs, color='khaki', alpha=0.3, label='Warm-up Phase')

        # 데이터 플로팅 (가장 중요한 Boundary Loss 기준)
        for model_name, hist in histories.items():
            if model_name in styles:
                s = styles[model_name]
                ax.plot(hist['boundary_loss'], color=s['color'], linestyle=s['ls'], 
                        linewidth=s['lw'], label=s['label'])

        ax.set_yscale('log')
        ax.set_xlabel("Epochs", fontsize=12, fontweight='bold')
        ax.set_ylabel("Boundary Focal Loss (Log Scale)", fontsize=12, fontweight='bold')
        ax.set_title("Convergence of Boundary Loss across Color Spaces and Activations", fontsize=14)
        
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc='upper right', frameon=True, shadow=True)

        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches='tight', format='pdf')
        plt.close()