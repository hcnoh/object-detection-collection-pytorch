import torch

from torch.nn import Module, Sequential, Conv2d, MaxPool2d

from config import DEVICE


class Darknet19Backbone(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.w_in = 416
        self.h_in = 416

        self.net1 = Sequential(
            Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=[3, 3],
                padding="same",
            ),
            MaxPool2d(
                kernel_size=[2, 2],
                stride=2,
            ),
        )

        self.net2 = Sequential(
            Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=[3, 3],
                padding="same",
            ),
            MaxPool2d(
                kernel_size=[2, 2],
                stride=2,
            ),
        )

        self.net3 = Sequential(
            Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=[3, 3],
                padding="same",
            ),
            Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=[1, 1],
                padding="same",
            ),
            Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=[3, 3],
                padding="same",
            ),
            MaxPool2d(
                kernel_size=[2, 2],
                stride=2,
            ),
        )

        self.net4 = Sequential(
            Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=[3, 3],
                padding="same",
            ),
            Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=[1, 1],
                padding="same",
            ),
            Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=[3, 3],
                padding="same",
            ),
            MaxPool2d(
                kernel_size=[2, 2],
                stride=2,
            ),
        )

        self.net5 = Sequential(
            Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=[3, 3],
                padding="same",
            ),
            Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=[1, 1],
                padding="same",
            ),
            Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=[3, 3],
                padding="same",
            ),
            Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=[1, 1],
                padding="same",
            ),
            Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=[3, 3],
                padding="same",
            ),
            MaxPool2d(
                kernel_size=[2, 2],
                stride=2,
            ),
        )

        self.net6 = Sequential(
            Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=[3, 3],
                padding="same",
            ),
            Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=[1, 1],
                padding="same",
            ),
            Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=[3, 3],
                padding="same",
            ),
            Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=[1, 1],
                padding="same",
            ),
            Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=[3, 3],
                padding="same",
            ),
        )

        self.net = Sequential(
            self.net1,
            self.net2,
            self.net3,
            self.net4,
            self.net5,
            self.net6,
        )

    def normalize(self, x):
        x = (
            (
                x / 255 -
                torch.tensor([[[[0.485, 0.456, 0.406]]]]).float().to(DEVICE)
            ) /
            torch.tensor([[[[0.229, 0.224, 0.225]]]]).float().to(DEVICE)
        )

        # x: [N, C, H, W]
        x = x.permute(0, 3, 1, 2)

        return x
