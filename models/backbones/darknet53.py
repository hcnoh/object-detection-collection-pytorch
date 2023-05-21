import torch

from torch.nn import (
    Module,
    ModuleList,
    Sequential,
    Conv2d,
    BatchNorm2d,
    LeakyReLU,
)

from config import DEVICE


class ConvLayer(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        leaky_relu=0.1
    ) -> None:
        super().__init__()

        self.net = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
            ),
            BatchNorm2d(out_channels),
            LeakyReLU(leaky_relu),
        )

    def forward(self, x):
        y = self.net(x)

        return y


class ConvAndResidualLayer(Module):
    def __init__(
        self,
        in_channels,
        net1_out_channels,
        net1_kernel_size,
        net1_stride,
        net2_out_channels,
        net2_kernel_size,
        net2_stride,
        leaky_relu=0.1,
    ) -> None:
        super().__init__()

        self.net1 = ConvLayer(
            in_channels,
            net1_out_channels,
            net1_kernel_size,
            net1_stride,
            leaky_relu,
        )

        self.net2 = ConvLayer(
            net1_out_channels,
            net2_out_channels,
            net2_kernel_size,
            net2_stride,
            leaky_relu,
        )

    def forward(self, x):
        h = self.net1(x)
        h = self.net2(h)
        y = h + x

        return y


class RepeatedConvAndResidualLayer(Module):
    def __init__(
        self,
        in_channels,
        net1_out_channels,
        net1_kernel_size,
        net1_stride,
        net2_out_channels,
        net2_kernel_size,
        net2_stride,
        repeat,
        leaky_relu=0.1,
    ) -> None:
        super().__init__()

        in_channel_list = (
            [in_channels] + [net2_out_channels for _ in range(repeat - 1)]
        )

        self.nets = ModuleList([
            ConvAndResidualLayer(
                ch,
                net1_out_channels,
                net1_kernel_size,
                net1_stride,
                net2_out_channels,
                net2_kernel_size,
                net2_stride,
                leaky_relu,
            )
            for ch in in_channel_list
        ])

    def forward(self, x):
        h = x

        for net in self.nets:
            h = net(h)
        y = h

        return y


class Darknet53Backbone(Module):
    def __init__(self) -> None:
        super().__init__()

        self.net1 = ConvLayer(
            in_channels=3,
            out_channels=32,
            kernel_size=[3, 3],
            stride=1,
        )

        self.net2 = ConvLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=[3, 3],
            stride=2,
        )

        self.net3 = RepeatedConvAndResidualLayer(
            in_channels=64,
            net1_out_channels=32,
            net1_kernel_size=[1, 1],
            net1_stride=1,
            net2_out_channels=64,
            net2_kernel_size=[3, 3],
            net2_stride=1,
            repeat=1,
        )

        self.net4 = ConvLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=[3, 3],
            stride=2,
        )

        self.net5 = RepeatedConvAndResidualLayer(
            in_channels=128,
            net1_out_channels=64,
            net1_kernel_size=[1, 1],
            net1_stride=1,
            net2_out_channels=128,
            net2_kernel_size=[3, 3],
            net2_stride=1,
            repeat=2,
        )

        self.net6 = ConvLayer(
            in_channels=128,
            out_channels=256,
            kernel_size=[3, 3],
            stride=2,
        )

        self.net7 = RepeatedConvAndResidualLayer(
            in_channels=256,
            net1_out_channels=128,
            net1_kernel_size=[1, 1],
            net1_stride=1,
            net2_out_channels=256,
            net2_kernel_size=[3, 3],
            net2_stride=1,
            repeat=8,
        )

        self.net8 = ConvLayer(
            in_channels=256,
            out_channels=512,
            kernel_size=[3, 3],
            stride=2,
        )

        self.net9 = RepeatedConvAndResidualLayer(
            in_channels=512,
            net1_out_channels=256,
            net1_kernel_size=[1, 1],
            net1_stride=1,
            net2_out_channels=512,
            net2_kernel_size=[3, 3],
            net2_stride=1,
            repeat=8,
        )

        self.net10 = ConvLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=[3, 3],
            stride=2,
        )

        self.net11 = RepeatedConvAndResidualLayer(
            in_channels=1024,
            net1_out_channels=512,
            net1_kernel_size=[1, 1],
            net1_stride=1,
            net2_out_channels=1024,
            net2_kernel_size=[3, 3],
            net2_stride=1,
            repeat=4,
        )

    def normalize(self, x):
        '''
            Args:
                x:
                    - the input image whose type is FloatTensor
                    - [batch_size, height, width, rgb]
        '''
        x = (
            (
                x / 255 -
                torch.tensor([0.485, 0.456, 0.406]).float().to(DEVICE)
            ) /
            torch.tensor([0.229, 0.224, 0.225]).float().to(DEVICE)
        )

        '''x: [batch_size, rgb, height, width]'''
        x = x.permute(0, 3, 1, 2)

        return x
