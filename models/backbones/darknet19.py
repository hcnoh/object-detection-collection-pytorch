import torch

from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    MaxPool2d,
    BatchNorm2d,
    LeakyReLU,
)

from config import DEVICE


class Darknet19Backbone(Module):
    def __init__(self) -> None:
        super().__init__()

        # self.w_in = 416
        # self.h_in = 416

        self.net1 = Sequential(
            Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(32),
            LeakyReLU(0.1),
        )

        self.net2 = Sequential(
            MaxPool2d(
                kernel_size=[2, 2],
                stride=2,
            ),
            Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(64),
            LeakyReLU(0.1),
        )

        self.net3 = Sequential(
            MaxPool2d(
                kernel_size=[2, 2],
                stride=2,
            ),
            Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(128),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=[1, 1],
                padding="same",
            ),
            BatchNorm2d(64),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(128),
            LeakyReLU(0.1),
        )

        self.net4 = Sequential(
            MaxPool2d(
                kernel_size=[2, 2],
                stride=2,
            ),
            Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(256),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=[1, 1],
                padding="same",
            ),
            BatchNorm2d(128),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(256),
            LeakyReLU(0.1),
        )

        self.net5 = Sequential(
            MaxPool2d(
                kernel_size=[2, 2],
                stride=2,
            ),
            Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=[1, 1],
                padding="same",
            ),
            BatchNorm2d(256),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=[1, 1],
                padding="same",
            ),
            BatchNorm2d(256),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(512),
            LeakyReLU(0.1),
        )

        self.net6 = Sequential(
            MaxPool2d(
                kernel_size=[2, 2],
                stride=2,
            ),
            Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=[1, 1],
                padding="same",
            ),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=[1, 1],
                padding="same",
            ),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
        )

        self.net7 = Sequential(
            Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
        )

    def forward(self, x):
        '''
            Args:
                x:
                    - the input image whose type is FloatTensor
                    - [N, H, W, C] = [N, 416, 416, 3]

            Returns:
                y:
                    - [N, output_dim, 13, 13]
        '''
        N, H, W, _ = x.shape

        # x: [N, C, H, W] = [N, 3, 416, 416]
        x = self.normalize(x)

        # h: [N, 32, 416, 416]
        h = self.net1(x)

        # h: [N, 64, 208, 208]
        h = self.net2(h)

        # h: [N, 128, 104, 104]
        h = self.net3(h)

        # h: [N, 256, 52, 52]
        h = self.net4(h)

        # h: [N, 512, 26, 26]
        h = self.net5(h)

        # h: [N, 1024, 13, 13]
        h = self.net6(h)

        # y: [N, 1024, 13, 13]
        y = self.net7(h)

        return y

    def normalize(self, x):
        '''
            Args:
                x:
                    - the input image whose type is FloatTensor
                    - [N, H, W, C]
        '''
        x = (
            (
                x / 255 -
                torch.tensor([0.485, 0.456, 0.406]).float().to(DEVICE)
            ) /
            torch.tensor([0.229, 0.224, 0.225]).float().to(DEVICE)
        )

        # x: [N, C, H, W]
        x = x.permute(0, 3, 1, 2)

        return x
