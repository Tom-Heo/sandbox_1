import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision import tv_tensors

class OxfordPetDataset(Dataset):
    """
    [Oxford-IIIT Pet Segmentation Dataset]
    - 입력 이미지(Image): [-1, 1] 범위의 Float 텐서
    - 정답 마스크(Mask): {0(전경), 1(배경), 2(경계선)}의 Long 텐서
    """
    def __init__(self, root: str = './data', split: str = 'trainval', image_size: int = 256):
        super().__init__()
        self.split = split
        self.image_size = image_size
        
        # 원본 데이터셋 로드 (다운로드 자동 처리)
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True
        )
        
        self.transforms = self._build_transforms()

    def _build_transforms(self):
        """[공간 증강 및 정규화 파이프라인]
        v2 API는 tv_tensors.Mask 타입이 감지되면 어떠한 공간 변형(Resize, Affine)에서도
        자동으로 'Nearest Neighbor' 보간법을 강제하여 유령 클래스(예: 1.5) 생성을 차단합니다.
        """
        common_transforms =[
            v2.Resize((self.image_size, self.image_size), antialias=True),
        ]

        if self.split == 'trainval':
            # 학습용: 공간적 증강 추가 (색상 증강은 OklabP 변환 기하학을 파괴하므로 제외)
            common_transforms.extend([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])

        common_transforms.extend([
            # 1. Image는 [0, 1] Float으로, Mask는 정수형(Long)으로 스케일링/캐스팅
            v2.ToDtype({
                tv_tensors.Image: torch.float32, 
                tv_tensors.Mask: torch.long
            }, scale=True),
            
            # 2. Image만 [-1, 1]로 정규화 (sRGB와 OklabP의 공정한 출발선)
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        return v2.Compose(common_transforms)

    def __getitem__(self, idx):
        img_pil, mask_pil = self.dataset[idx]

        # 1. v2 인식용 타입 매핑 (이 선언 하나로 보간법 함정이 완벽히 방어됨)
        img = tv_tensors.Image(img_pil)
        mask = tv_tensors.Mask(mask_pil)

        # 2. Joint Transform 적용 (이미지와 마스크가 동일한 각도/비율로 동기화되어 변형됨)
        img, mask = self.transforms(img, mask)

        # 3. Trimap 클래스 매핑: {1, 2, 3} -> {0, 1, 2}
        # - 0: Pet (전경)
        # - 1: Background (배경)
        # - 2: Boundary (경계선)
        mask = mask - 1

        # 4. Loss 계산을 위해 마스크의 채널 차원 (1, H, W) -> (H, W) 축소
        mask = mask.squeeze(0)

        return img, mask

    def __len__(self):
        return len(self.dataset)

# ---------------------------------------------------------
#[데이터로더 생성 팩토리 함수]
def get_dataloaders(root: str = './data', batch_size: int = 16, image_size: int = 256):
    """
    재현성을 위해 worker_init_fn이나 Generator 설정은 
    메인 스크립트(utils/seed.py)에서 전역 시드와 함께 통제하는 것을 권장합니다.
    """
    train_ds = OxfordPetDataset(root, split='trainval', image_size=image_size)
    val_ds = OxfordPetDataset(root, split='test', image_size=image_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    return train_loader, val_loader