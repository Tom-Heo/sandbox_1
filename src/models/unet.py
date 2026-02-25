import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 1. Activation Factory & Block
# ---------------------------------------------------------
class DoubleConv(nn.Module):
    """[모듈화된 합성곱 블록]
    어떤 Activation이 들어와도 완벽하게 소화하도록 'act_builder' 함수를 주입받습니다.
    """

    def __init__(self, in_channels, out_channels, act_builder):
        super().__init__()
        # Conv 레이어에서는 bias를 False로 둡니다. (뒤에 BatchNorm이 오기 때문)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act_builder(
            out_channels
        )  # 채널 수를 동적으로 주입하여 Activation 생성

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act_builder(out_channels)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


# ---------------------------------------------------------
# 2. Base U-Net Architecture
# ---------------------------------------------------------
class CustomUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=3,
        features=[64, 128, 256, 512],
        act_builder=None,
    ):
        super().__init__()

        # 기본값은 ReLU (채널 인자를 무시하도록 lambda 처리)
        if act_builder is None:
            act_builder = lambda c: nn.ReLU(inplace=True)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # [Encoder]
        in_ch = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_ch, feature, act_builder))
            in_ch = feature

        # [Bottleneck]
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, act_builder)

        # [Decoder]
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature, act_builder))

        # [Final Classifier] (활성화 함수 없음 -> 손실 함수에서 Softmax/Logit 처리)
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

        # 통제 변인: Xavier 초기화 적용
        self._initialize_weights()

    def _initialize_weights(self):
        """
        HeLU2d의 분산 보존 특성에 맞춰 모든 Conv 레이어를 Kaming He Initialization 방식으로 초기화합니다.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        skip_connections = []

        # Down
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Up
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]

            # 해상도 불일치 시 Bilinear 보간으로 맞춤 (안전장치)
            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)


# ---------------------------------------------------------
# 3. Experiment Wrapper (실험 통제용 껍데기 모델)
# ---------------------------------------------------------
class PetSegmentationModel(nn.Module):
    """
    이 래퍼 클래스 하나로 4가지 실험(sRGB/OklabP x ReLU/HeLU2d)을 완벽히 통제합니다.
    """

    def __init__(self, use_oklab: bool = False, use_helu: bool = False):
        super().__init__()
        self.use_oklab = use_oklab

        # 1. Activation Builder 설정
        if use_helu:
            # HeLU2d는 채널 수가 필요하므로 lambda로 동적 생성
            from src.models.activations import Heo  # (HeLU2d 코드가 있는 위치)

            act_builder = lambda c: Heo.HeLU2d(channels=c)
        else:
            act_builder = lambda c: nn.ReLU(inplace=True)

        # 2. 색공간 변환기 (OklabP)
        if self.use_oklab:
            from src.models.colors import Palette  # (작성하신 Palette 코드가 있는 위치)

            self.color_converter = Palette.sRGBtoOklabP()

        # 3. U-Net 본체
        self.unet = CustomUNet(in_channels=3, num_classes=3, act_builder=act_builder)

    def forward(self, x):
        # x는 Dataset에서 넘어온 [-1, 1] 범위의 sRGB 텐서입니다.

        if self.use_oklab:
            # 짜오신 Palette.sRGBtoOklabP는[0, 1] 입력을 기대하므로,
            # 일시적으로 [-1, 1]을[0, 1]로 스케일링한 뒤 넘깁니다.
            x = (x + 1.0) / 2.0

            # OklabP를 통과하면 다시 논리적인[-1, 1] 스케일의 Lp, ap, bp 텐서가 나옵니다.
            x = self.color_converter(x)

        return self.unet(x)
