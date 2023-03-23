import torch

from torch.nn import Module

from config import DEVICE


class GoogLeNetBackbone(Module):
    def __init__(self) -> None:
        super().__init__()

        self.backbone_model = torch.hub.load(
            'pytorch/vision:v0.10.0', 'googlenet', pretrained=True
        )

        self.w_in = 224
        self.h_in = 224

        self.output_shape = [7, 7, 1024]

    def forward(self, x):
        '''
            Args:
                x: The input image whose type is FloatTensor with the shape of
                [N, H=224, W=224, C=3].
        '''

        # x: [N, C, H, W]
        x = self.normalize(x)

        # h: [N, 1024, 7, 7]
        h = self.backbone_model.conv1(x)
        h = self.backbone_model.maxpool1(h)
        h = self.backbone_model.conv2(h)
        h = self.backbone_model.conv3(h)
        h = self.backbone_model.maxpool2(h)
        h = self.backbone_model.inception3a(h)
        h = self.backbone_model.inception3b(h)
        h = self.backbone_model.maxpool3(h)
        h = self.backbone_model.inception4a(h)
        h = self.backbone_model.inception4b(h)
        h = self.backbone_model.inception4c(h)
        h = self.backbone_model.inception4d(h)
        h = self.backbone_model.inception4e(h)
        h = self.backbone_model.maxpool4(h)
        h = self.backbone_model.inception5a(h)
        h = self.backbone_model.inception5b(h)

        # y: [N, 7, 7, 1024]
        y = h.permute(0, 2, 3, 1)

        return y

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
