import os
import pickle

import numpy as np
import torch
import torch.cuda
import torch.backends.mps
import albumentations
import albumentations.pytorch

from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, LeakyReLU
from torch.nn.functional import one_hot
from torch.optim import SGD

from config import DEVICE
from models.backbones.darknet19 import Darknet19Backbone
from models.utils import get_iou, get_aps, nms, cummax


TRANSFORM = albumentations.Compose(
    [
        albumentations.RandomScale(scale_limit=(-0.2, 0.2), p=0.5),
        albumentations.Affine(translate_percent=(-0.2, 0.2), p=0.5),
        albumentations.Affine(rotate=(-45, 45), p=0.5),
        albumentations.OneOf(
            [
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
            ],
            p=0.5,
        ),
        albumentations.ChannelShuffle(p=0.5),
        albumentations.HueSaturationValue(p=0.5),
    ],
    bbox_params=albumentations.BboxParams(
        format="pascal_voc", label_fields=["labels"]
    ),
)


class YOLOv2(Module):
    def __init__(
        self,
        cls_list,
        cls2idx,
    ) -> None:
        super().__init__()

        self.anchor_box_size_list = [
            (1.3221, 1.73145),
            (3.19275, 4.00944),
            (5.05587, 8.09892),
            (9.47112, 4.84053),
            (11.2364, 10.0071),
        ]
        self.num_anchor_box = len(self.anchor_box_size_list)

        self.anchor_box_width_list = (
            torch.tensor([b[0] for b in self.anchor_box_size_list]).to(DEVICE)
        )
        self.anchor_box_height_list = (
            torch.tensor([b[1] for b in self.anchor_box_size_list]).to(DEVICE)
        )

        self.cls_list = cls_list
        self.cls2idx = cls2idx

        self.num_cls = len(self.cls_list)

        self.head_output_dim = self.num_anchor_box * (5 + self.num_cls)

        self.backbone_model = Darknet19Backbone()

        self.head_model = Sequential(
            Conv2d(
                in_channels=3072,
                out_channels=1024,
                kernel_size=[3, 3],
                padding="same",
            ),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(
                in_channels=1024,
                out_channels=self.head_output_dim,
                kernel_size=[1, 1],
                padding="same",
            ),
            BatchNorm2d(self.head_output_dim),
        )

    def backbone(self, x):
        '''
            Args:
                x:
                    - the input image whose type is FloatTensor
                    - [batch_size, height, width, rgb]
                        = [batch_size, 416, 416, 3]

            Returns:
                h1:
                    - [
                        batch_size, 512,
                        num_grid_cell_in_height1,
                        num_grid_cell_in_width1
                    ]
                        = [batch_size, 512, 26, 26]
                h2:
                    - [
                        batch_size, 1024,
                        num_grid_cell_in_height2,
                        num_grid_cell_in_width2
                    ]
                        = [batch_size, 1024, 13, 13]
        '''

        '''
        x:
            - [batch_size, rgb, height, width] = [batch_size, 3, 416, 416]
        '''
        x = self.backbone_model.normalize(x)

        '''
        h:
            - [batch_size, 32, height, width] = [batch_size, 32, 416, 416]
        '''
        h = self.backbone_model.net1(x)

        '''
        h:
            - [batch_size, 64, height2, width2]
                = [batch_size, 64, height // 2, width // 2]
                = [batch_size, 64, 208, 208]
        '''
        h = self.backbone_model.net2(h)

        '''
        h:
            - [batch_size, 128, height3, width3]
                = [batch_size, 128, height2 // 2, width2 // 2]
                = [batch_size, 128, 104, 104]
        '''
        h = self.backbone_model.net3(h)

        '''
        h:
            - [batch_size, 256, height4, width4]
                = [batch_size, 256, height3 // 2, width3 // 2]
                = [batch_size, 256, 52, 52]
        '''
        h = self.backbone_model.net4(h)

        '''
        h1:
            - [
                batch_size,
                512,
                num_grid_cell_in_height1,
                num_grid_cell_in_width1,
            ]
                = [batch_size, 512, height4 // 2, width4 // 2]
                = [batch_size, 512, 26, 26]
        '''
        h1 = self.backbone_model.net5(h)

        '''
        h2:
            - [
                batch_size,
                1024,
                num_grid_cell_in_height2,
                num_grid_cell_in_width2,
            ]
                = [batch_size, 1024, height5 // 2, width5 // 2]
                = [batch_size, 1024, 13, 13]
        '''
        h2 = self.backbone_model.net6(h1)

        '''
         h2:
            - [
                batch_size,
                1024,
                num_grid_cell_in_height2,
                num_grid_cell_in_width2,
            ]
                = [batch_size, 1024, 13, 13]
        '''
        h2 = self.backbone_model.net7(h2)

        return h1, h2

    def neck(self, h1, h2):
        '''
            Args:
                h1:
                    - [
                        batch_size,
                        512,
                        num_grid_cell_in_height1,
                        num_grid_cell_in_width1
                    ]
                        = [batch_size, 512, 26, 26]
                h2:
                    - [
                        batch_size,
                        1024,
                        num_grid_cell_in_height2,
                        num_grid_cell_in_width2
                    ]
                        = [batch_size, 1024, 13, 13]

            Returns:
                h:
                    - [
                        batch_size,
                        3072,
                        num_grid_cell_in_height2,
                        num_grid_cell_in_width2,
                    ]
                        = [batch_size, 3072, 13, 13]
        '''
        _, _, num_grid_cell_in_height1, num_grid_cell_in_width1 = h1.shape
        _, _, num_grid_cell_in_height2, num_grid_cell_in_width2 = h2.shape

        assert (
            num_grid_cell_in_height2 == num_grid_cell_in_height1 // 2 and
            num_grid_cell_in_width2 == num_grid_cell_in_width1 // 2
        )

        '''
        h1:
            - [
                batch_size,
                512,
                num_grid_cell_in_height1,
                num_grid_cell_in_width1,
            ]
            ->
            [
                2,
                batch_size,
                512,
                num_grid_cell_in_height1,
                num_grid_cell_in_width2,
            ]
            ->
            [
                batch_size,
                1024,
                num_grid_cell_in_height1,
                num_grid_cell_in_width2,
            ]
        '''
        h1 = torch.cat(
            [
                h1[
                    :, :, :,
                    i * num_grid_cell_in_width2:
                    i * num_grid_cell_in_width2 + num_grid_cell_in_width2
                ]
                for i in range(2)
            ],
            dim=1,
        )

        '''
        h1:
            - [
                batch_size,
                1024,
                num_grid_cell_in_height1,
                num_grid_cell_in_width2,
            ]
            ->
            [
                2,
                batch_size,
                1024,
                num_grid_cell_in_height2,
                num_grid_cell_in_width2,
            ]
            ->
            [
                batch_size,
                2048,
                num_grid_cell_in_height2,
                num_grid_cell_in_width2,
            ]
        '''
        h1 = torch.cat(
            [
                h1[
                    :, :,
                    i * num_grid_cell_in_height2:
                    i * num_grid_cell_in_height2 + num_grid_cell_in_height2,
                    :
                ]
                for i in range(2)
            ],
            dim=1,
        )

        '''
        h:
            - [
                batch_size,
                3072,
                num_grid_cell_in_height2,
                num_grid_cell_in_width2,
            ]
        '''
        h = torch.cat([h1, h2], dim=1)

        return h

    def head(self, h):
        '''
            Args:
                h:
                    - [batch_size, 3072, H, W]

            Returns:
                y:
                    - [batch_size, H, W, output_dim]
        '''
        num_anchor_box = self.num_anchor_box
        num_cls = self.num_cls

        '''
        anchor_box_width_list, anchor_box_height_list:
            - [1, 1, 1, num_anchor_box]
        '''
        pw = (
            self.anchor_box_width_list.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        ph = (
            self.anchor_box_height_list.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        '''
        y:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box * (5 + num_cls),
            ]
        '''
        y = self.head_model(h).permute(0, 2, 3, 1)

        '''
        tx, ty -> sigmoid(tx), sigmoid(ty)
        '''
        y[..., 0::5 + num_cls] = torch.sigmoid(y[..., 0::5 + num_cls])
        y[..., 1::5 + num_cls] = torch.sigmoid(y[..., 1::5 + num_cls])

        '''
        tw, th -> pw * exp(tw), ph * exp(th)
        '''
        y[..., 2::5 + num_cls] = pw * torch.exp(y[..., 2::5 + num_cls])
        y[..., 3::5 + num_cls] = ph * torch.exp(y[..., 3::5 + num_cls])

        '''
        to -> sigmoid(to)
        '''
        y[..., 4::5 + num_cls] = torch.sigmoid(y[..., 4::5 + num_cls])

        '''
        cond_cls_prob
        '''
        for i in range(num_anchor_box):
            y[..., i * (num_cls + 5) + 5:i * (num_cls + 5) + 5 + num_cls] = (
                torch.softmax(
                    y[
                        ...,
                        i * (num_cls + 5) + 5:i * (num_cls + 5) + 5 + num_cls
                    ],
                    dim=-1
                )
            )

        return y

    def forward(self, x):
        '''
            Args:
                x:
                    - the input image whose type is FloatTensor
                    - [batch_size, height, width, rgb]
                        = [batch_size, 416, 416, 3]

            Returns:
                y:
                    - [
                        batch_size,
                        num_grid_cell_in_height,
                        num_grid_cell_width,
                        output_dim
                    ]
        '''

        '''
        h1:
            - [
                batch_size,
                512,
                num_grid_cell_in_height1,
                num_grid_cell_width1,
            ]
        h2:
            - [
                batch_size,
                1024,
                num_grid_cell_in_height2,
                num_grid_cell_width2,
            ]
        '''
        h1, h2 = self.backbone(x)

        '''
        h:
            - [
                batch_size,
                3072,
                num_grid_cell_in_height,
                num_grid_cell_width,
            ]
                = [
                    batch_size,
                    3072,
                    num_grid_cell_in_height2,
                    num_grid_cell_width2,
                ]
        '''
        h = self.neck(h1, h2)

        '''
        y:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_width,
                num_anchor_box * (5 + num_cls),
            ]
        '''
        y = self.head(h)

        return y

    def predict(
        self,
        x_batch,
    ):
        '''
            Args:
                x_batch:
                    - the input image batch whose type is FloatTensor
                    - [batch_size, height, width, rgb]
        '''
        _, height, width, _ = x_batch.shape

        '''
        y_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_width,
                num_anchor_box * (5 + num_cls),
            ]
        '''
        y_pred_batch = self(x_batch)

        _, num_grid_cell_height, num_grid_cell_width, _ = y_pred_batch.shape
        num_cls = self.num_cls

        '''
        bx_norm_pred_batch, by_norm_pred_batch,
        bw_pred_batch, bh_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
            ]
        bbox_norm_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
                4,
            ]
        '''
        bx_norm_pred_batch = y_pred_batch[..., 0::5 + num_cls]
        by_norm_pred_batch = y_pred_batch[..., 1::5 + num_cls]
        bw_pred_batch = y_pred_batch[..., 2::5 + num_cls]
        bh_pred_batch = y_pred_batch[..., 3::5 + num_cls]

        bbox_norm_pred_batch = torch.stack(
            [
                bx_norm_pred_batch,
                by_norm_pred_batch,
                bw_pred_batch,
                bh_pred_batch,
            ],
            dim=-1
        )

        '''
        cy_batch:
            - [1, num_grid_cell_height, 1, 1]
        cx_batch:
            - [1, 1, num_grid_cell_width, 1]
        '''
        cy_batch = (
            torch.arange(num_grid_cell_height).to(DEVICE)
            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            .detach()
        )
        cx_batch = (
            torch.arange(num_grid_cell_width).to(DEVICE)
            .unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            .detach()
        )

        '''
        bx_pred_batch, by_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
            ]
        '''
        bx_pred_batch = (
            cx_batch + bx_norm_pred_batch
        )
        by_pred_batch = (
            cy_batch + by_norm_pred_batch
        )

        '''
        x1_pred_batch, y1_pred_batch,
        x2_pred_batch, y2_pred_batch,
        x1_norm_pred_batch, y1_norm_pred_batch,
        x2_norm_pred_batch, y2_norm_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
            ]
        bbox_coord_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
                4,
            ]
        '''
        x1_norm_pred_batch = bx_pred_batch - (bw_pred_batch / 2)
        y1_norm_pred_batch = by_pred_batch - (bh_pred_batch / 2)
        x2_norm_pred_batch = bx_pred_batch + (bw_pred_batch / 2)
        y2_norm_pred_batch = by_pred_batch + (bh_pred_batch / 2)

        grid_cell_height = height / num_grid_cell_height
        grid_cell_width = width / num_grid_cell_width

        x1_pred_batch = x1_norm_pred_batch * grid_cell_width
        y1_pred_batch = y1_norm_pred_batch * grid_cell_height
        x2_pred_batch = x2_norm_pred_batch * grid_cell_width
        y2_pred_batch = y2_norm_pred_batch * grid_cell_height

        bbox_coord_pred_batch = torch.stack(
            [
                x1_pred_batch,
                y1_pred_batch,
                x2_pred_batch,
                y2_pred_batch,
            ],
            dim=-1
        )

        '''
        conf_score_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
            ]
        cond_cls_prob_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
                num_cls,
            ]
        '''
        conf_score_pred_batch = y_pred_batch[..., 4::5 + num_cls]
        cond_cls_prob_pred_batch = torch.stack(
            [y_pred_batch[..., i::5 + num_cls] for i in range(5, 5 + num_cls)],
            dim=-1
        )

        '''
        cls_spec_conf_score_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
                num_cls,
            ]
        '''
        cls_spec_conf_score_pred_batch = (
            cond_cls_prob_pred_batch *
            conf_score_pred_batch.unsqueeze(-1)
        )

        return (
            bbox_norm_pred_batch,
            bbox_coord_pred_batch,
            conf_score_pred_batch,
            cond_cls_prob_pred_batch,
            cls_spec_conf_score_pred_batch,
        )

    def detect(
        self,
        img,
        conf_score_thre=0.9,
        iou_thre=0.5,
    ):
        '''
            Args:
                img:
                    - the input image whose type is NDArray
                    - [height, width, rgb]
        '''

        self.eval()

        '''x_batch: [batch_size, height, width, rgb]'''
        x_batch = torch.tensor([img]).to(DEVICE)

        '''
        bbox_coord_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
                4,
            ]
        conf_score_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
            ]
        cls_spec_conf_score_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
                num_cls,
            ]
        '''
        (
            _,
            bbox_coord_pred_batch,
            conf_score_pred_batch,
            _,
            cls_spec_conf_score_pred_batch,
        ) = self.predict(x_batch)

        '''
        bbox_coord_pred_batch: [num_bbox, 4]
        conf_score_pred_batch: [num_bbox]
        cls_spec_conf_score_pred_batch: [num_bbox, num_cls]
        '''
        (
            bbox_coord_pred_batch,
            conf_score_pred_batch,
            cls_spec_conf_score_pred_batch,
        ) = nms(
            bbox_coord_pred_batch,
            conf_score_pred_batch,
            cls_spec_conf_score_pred_batch,
            conf_score_thre,
            iou_thre,
        )

        bbox_coord_pred_batch = bbox_coord_pred_batch.detach().cpu().numpy()
        conf_score_pred_batch = conf_score_pred_batch.detach().cpu().numpy()
        cls_spec_conf_score_pred_batch = (
            cls_spec_conf_score_pred_batch.detach().cpu().numpy()
        )

        max_cls_spec_conf_score_pred_batch = (
            cls_spec_conf_score_pred_batch.max(-1)
        )
        argmax_cls_spec_conf_score_pred_batch = (
            cls_spec_conf_score_pred_batch.argmax(-1)
        )

        annot_pred = {
            "bbox_list": bbox_coord_pred_batch.tolist(),
            "lbl_list": [
                self.cls_list[cls_idx]
                for cls_idx in argmax_cls_spec_conf_score_pred_batch
            ],
            "conf_score_list": conf_score_pred_batch.tolist(),
            "cls_spec_conf_score_list": (
                max_cls_spec_conf_score_pred_batch.tolist()
            ),
        }

        return annot_pred

    def get_loss(
        self,
        x_batch,
        bbox_norm_tgt_batch,
        bbox_coord_tgt_batch,
        cls_tgt_batch,
        obj_mask_batch,
        x_img_id_batch,
        bbox_img_id_batch,
        lambda_xy,
        lambda_wh,
        lambda_conf,
        lambda_noobj,
        lambda_cls,
    ):
        '''
            Args:
                x_batch:
                    - the input image batch whose type is FloatTensor
                    - [batch_size, height, width, rgb]
                bbox_norm_tgt_batch:
                    - the given normalized bounding box targets for all
                        bounding boxes
                    - [
                        num_bbox,
                        num_grid_cell_in_height,
                        num_grid_cell_in_width,
                        4
                    ]
                bbox_coord_tgt_batch:
                    - the given coordinates for the all bounding boxes
                    - [
                        num_bbox,
                        num_grid_cell_in_height,
                        num_grid_cell_in_width,
                        4
                    ]
                cls_tgt_batch:
                    - the given class targets as onehot vectors
                    - [
                        num_bbox,
                        num_grid_cell_in_height,
                        num_grid_cell_in_width,
                        num_cls
                    ]
                obj_mask_batch:
                    - the mask to indicate the object is in each grid cell
                    - [
                        num_bbox,
                        num_grid_cell_in_height,
                        num_grid_cell_in_width
                    ]
                x_img_id_batch:
                    - the image IDs for the given input
                    - [batch_size]
                bbox_img_id_batch:
                    - the image IDs for the given bounding boxes
                    - [num_bbox]
        '''
        eps = 1e-6

        num_anchor_box = self.num_anchor_box

        '''
        pw, ph:
            - [1, 1, 1, num_anchor_box]
        '''
        pw = self.anchor_box_width_list.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        ph = self.anchor_box_height_list.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        '''
        bbox_img_id_to_x_img_id_mapper:
            - [num_bbox, batch_size] -> [num_bbox]
        '''
        bbox_img_id_to_x_img_id_mapper = (
            (
                bbox_img_id_batch.unsqueeze(-1) ==
                x_img_id_batch.unsqueeze(0)
            ).long()
            .argmax(-1)
        )

        '''
        bbox_norm_pred_batch, bbox_coord_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
                4,
            ] ->
            [
                num_bbox,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
                4,
            ]
        conf_score_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
            ] ->
            [
                num_bbox,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
            ]
        cond_cls_prob_pred_batch, cls_spec_conf_score_pred_batch:
            - [
                batch_size,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
                num_cls,
            ] ->
            [
                num_bbox,
                num_grid_cell_height,
                num_grid_cell_width,
                num_anchor_box,
                num_cls,
            ]
        '''
        (
            bbox_norm_pred_batch,
            bbox_coord_pred_batch,
            conf_score_pred_batch,
            cond_cls_prob_pred_batch,
            cls_spec_conf_score_pred_batch,
        ) = self.predict(x_batch)

        (
            bbox_norm_pred_batch,
            bbox_coord_pred_batch,
            conf_score_pred_batch,
            cond_cls_prob_pred_batch,
            cls_spec_conf_score_pred_batch,
        ) = (
            bbox_norm_pred_batch[bbox_img_id_to_x_img_id_mapper],
            bbox_coord_pred_batch[bbox_img_id_to_x_img_id_mapper],
            conf_score_pred_batch[bbox_img_id_to_x_img_id_mapper],
            cond_cls_prob_pred_batch[bbox_img_id_to_x_img_id_mapper],
            cls_spec_conf_score_pred_batch[bbox_img_id_to_x_img_id_mapper],
        )

        '''
        tx_pred_batch, ty_pred_batch, tw_pred_batch, th_pred_batch:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        '''
        tx_pred_batch = torch.logit(bbox_norm_pred_batch[..., 0], eps)
        ty_pred_batch = torch.logit(bbox_norm_pred_batch[..., 1], eps)
        tw_pred_batch = torch.log(bbox_norm_pred_batch[..., 2] / pw + eps)
        th_pred_batch = torch.log(bbox_norm_pred_batch[..., 3] / ph + eps)

        '''
        tx_tgt_batch, ty_tgt_batch, tw_tgt_batch, th_tgt_batch:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        '''
        tx_tgt_batch = torch.logit(
            bbox_norm_tgt_batch[..., 0].unsqueeze(-1).clamp(0., 1.), eps
        )
        ty_tgt_batch = torch.logit(
            bbox_norm_tgt_batch[..., 1].unsqueeze(-1).clamp(0., 1.), eps
        )
        tw_tgt_batch = torch.log(
            bbox_norm_tgt_batch[..., 2].unsqueeze(-1) / pw + eps
        )
        th_tgt_batch = torch.log(
            bbox_norm_tgt_batch[..., 3].unsqueeze(-1) / ph + eps
        )

        '''
        cls_tgt_batch:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                1,
                num_cls,
            ]
        '''
        cls_tgt_batch = cls_tgt_batch.unsqueeze(-2)

        '''
        obj_mask_batch:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                1,
            ]
        '''
        obj_mask_batch = obj_mask_batch.unsqueeze(-1)

        '''
        bbox_coord_tgt_batch:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                1,
                4,
            ]
        '''
        bbox_coord_tgt_batch = bbox_coord_tgt_batch.unsqueeze(-2)

        '''
        iou_batch:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        '''
        iou_batch = get_iou(
            bbox_coord_pred_batch,
            bbox_coord_tgt_batch,
        ).detach()

        '''
        responsible_mask_batch:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
            ] ->
            [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        not_responsible_mask_batch:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        '''
        _, responsible_mask_batch = (
            torch.max(iou_batch, dim=-1)
        )
        responsible_mask_batch = one_hot(
            responsible_mask_batch, num_anchor_box
        )
        responsible_mask_batch = (responsible_mask_batch * obj_mask_batch)

        not_responsible_mask_batch = (responsible_mask_batch != 1)

        responsible_mask_batch = responsible_mask_batch.bool()
        not_responsible_mask_batch = not_responsible_mask_batch.bool()

        '''
        loss_xy:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ] ->
            [num_bbox] -> []
        '''
        loss_xy = (
            (tx_tgt_batch - tx_pred_batch) ** 2 +
            (ty_tgt_batch - ty_pred_batch) ** 2
        )
        loss_xy = torch.masked_select(loss_xy, responsible_mask_batch)
        loss_xy = loss_xy.mean()

        '''
        loss_wh:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ] ->
            [num_bbox] -> []
        '''
        loss_wh = (
            (tw_tgt_batch - tw_pred_batch) ** 2 +
            (th_tgt_batch - th_pred_batch) ** 2
        )
        loss_wh = torch.masked_select(loss_wh, responsible_mask_batch)
        loss_wh = loss_wh.mean()

        '''
        loss_conf:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ] ->
            [num_bbox] -> []
        '''
        loss_conf = (iou_batch - conf_score_pred_batch) ** 2
        loss_conf = torch.masked_select(loss_conf, responsible_mask_batch)
        loss_conf = loss_conf.mean()

        '''
        loss_noobj:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ] ->
            [
                num_bbox *
                num_grid_cell_in_height *
                num_grid_cell_in_width *
                num_anchor_box -
                num_bbox
            ] ->
            []
        '''
        loss_noobj = (0 - conf_score_pred_batch) ** 2
        loss_noobj = torch.masked_select(
            loss_noobj, not_responsible_mask_batch
        )
        loss_noobj = loss_noobj.mean()

        '''
        loss_cls:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                num_cls,
            ] ->
            [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ] ->
            [num_bbox] -> []
        '''
        loss_cls = (cls_tgt_batch - cond_cls_prob_pred_batch) ** 2
        loss_cls = loss_cls.sum(-1)
        loss_cls = torch.masked_select(loss_cls, responsible_mask_batch)
        loss_cls = loss_cls.mean()

        '''
        loss:
            - []
        '''
        loss = (
            lambda_xy * loss_xy +
            lambda_wh * loss_wh +
            lambda_conf * loss_conf +
            lambda_noobj * loss_noobj +
            lambda_cls * loss_cls
        )

        return loss

    def run_one_epoch(
        self,
        epoch,
        data_loader,
        lambda_xy,
        lambda_wh,
        lambda_conf,
        lambda_noobj,
        lambda_cls,
        lr=None,
        train=True,
    ):
        loss_mean = []

        dataset_size = len(data_loader.dataset)
        progress_size = 0

        for batch in data_loader:
            '''
            x_batch_one_step:
                - [
                    batch_size,
                    num_grid_cell_in_height,
                    num_grid_cell_in_width,
                    3,
                ]
            bbox_norm_tgt_batch_one_step:
                - [
                    num_bbox,
                    num_grid_cell_in_height,
                    num_grid_cell_in_width,
                    4,
                ]
            bbox_coord_tgt_batch_one_step:
                - [
                    num_bbox,
                    num_grid_cell_in_height,
                    num_grid_cell_in_width,
                    4,
                ]
            cls_tgt_batch_one_step:
                - [
                    num_bbox,
                    num_grid_cell_in_height,
                    num_grid_cell_in_width,
                    num_cls,
                ]
            obj_mask_batch_one_step:
                - [
                    num_bbox,
                    num_grid_cell_in_height,
                    num_grid_cell_in_width,
                ]
            x_img_id_batch_one_step:
                - [batch_size]
            bbox_img_id_batch_one_step:
                - [num_bbox]
            '''
            (
                x_batch_one_step,
                bbox_norm_tgt_batch_one_step,
                bbox_coord_tgt_batch_one_step,
                cls_tgt_batch_one_step,
                obj_mask_batch_one_step,
                x_img_id_batch_one_step,
                bbox_img_id_batch_one_step,
            ) = batch

            batch_size = x_batch_one_step.shape[0]

            progress_size += batch_size

            if train:
                print(
                    "Epoch: {} --> Training: [{} / {}]"
                    .format(epoch, progress_size, dataset_size),
                    end="\r"
                )

                self.train()

            else:
                print(
                    "Epoch: {} --> Validation: [{} / {}]"
                    .format(epoch, progress_size, dataset_size),
                    end="\r"
                )

                self.eval()

            '''
            loss_one_step:
                - []
            '''
            loss_one_step = self.get_loss(
                x_batch_one_step,
                bbox_norm_tgt_batch_one_step,
                bbox_coord_tgt_batch_one_step,
                cls_tgt_batch_one_step,
                obj_mask_batch_one_step,
                x_img_id_batch_one_step,
                bbox_img_id_batch_one_step,
                lambda_xy,
                lambda_wh,
                lambda_conf,
                lambda_noobj,
                lambda_cls,
            )

            if train:
                if epoch == 1:
                    opt = SGD(
                        self.parameters(),
                        lr=lr / (10 ** (1 - (progress_size / dataset_size))),
                        momentum=0.9,
                        weight_decay=5e-4,
                    )

                else:
                    opt = SGD(
                        self.parameters(),
                        lr=lr,
                        momentum=0.9,
                        weight_decay=5e-4,
                    )

                opt.zero_grad()
                loss_one_step.backward()
                opt.step()

            loss_mean.append(loss_one_step.detach().cpu().numpy())

        loss_mean = np.mean(loss_mean)

        return loss_mean

    def train_model(
        self,
        train_loader,
        val_loader,
        learning_rate_list,
        num_epoch_list,
        lambda_xy,
        lambda_wh,
        lambda_conf,
        lambda_noobj,
        lambda_cls,
        ckpt_path,
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                val_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        cum_epoch = 0

        train_loss_mean_list = []
        val_loss_list = []

        min_val_loss = 1e+10

        self.transform = TRANSFORM

        for lr, num_epochs in zip(learning_rate_list, num_epoch_list):

            for epoch in range(1 + cum_epoch, num_epochs + 1 + cum_epoch):
                if epoch - 1 % 10 == 0:
                    self.resize = self.get_random_size_transform()

                train_loss_mean = self.run_one_epoch(
                    epoch,
                    train_loader,
                    lambda_xy,
                    lambda_wh,
                    lambda_conf,
                    lambda_noobj,
                    lambda_cls,
                    lr,
                    train=True,
                )
                val_loss = self.run_one_epoch(
                    epoch,
                    val_loader,
                    lambda_xy,
                    lambda_wh,
                    lambda_conf,
                    lambda_noobj,
                    lambda_cls,
                    train=False,
                )

                print(
                    (
                        "Epoch: {} --> " +
                        "Training: (" +
                        "Loss Mean: {}" +
                        ")    " +
                        "Validation: (" +
                        "Loss: {}" +
                        ")"
                    )
                    .format(
                        epoch,
                        train_loss_mean,
                        val_loss,
                    )
                )

                train_loss_mean_list.append(train_loss_mean)

                val_loss_list.append(val_loss)

                if val_loss < min_val_loss:
                    torch.save(
                        self.state_dict(),
                        os.path.join(
                            ckpt_path, "best_model.ckpt"
                        )
                    )
                    min_val_loss = val_loss

            cum_epoch += num_epochs

        torch.save(
            self.state_dict(),
            os.path.join(
                ckpt_path, "final_model.ckpt"
            )
        )

        with open(
            os.path.join(ckpt_path, "training_result.pkl"),
            "wb",
        ) as f:
            pickle.dump(
                {
                    "train_loss_mean_list": train_loss_mean_list,
                    "val_loss_list": val_loss_list,
                },
                f
            )

    def evaluate_model(
        self,
        dataset,
        ckpt_path,
        conf_score_thre=0.9,
        iou_thre=0.5,
        level_list=[.50, .55, .60, .65, .70, .75, .80, .85, .90, .95],
    ):
        eps = 1e-6

        '''level_list: [num_level]'''
        level_list = np.array(level_list)

        cls_spec_tp_list = {
            c: [] for c in self.cls_list
        }
        cls_spec_fp_list = {
            c: [] for c in self.cls_list
        }
        cls_spec_conf_score_list = {
            c: [] for c in self.cls_list
        }
        cls_spec_num_gt_list = {
            c: 0 for c in self.cls_list
        }

        dataset_size = len(dataset)
        progress_size = 0

        for _, img, annot in dataset:
            progress_size += 1
            if progress_size > 1000:
                break

            print(
                "Evaluation: [{} / {}]".format(progress_size, dataset_size),
                end="\r"
            )

            '''
            coord_batch: [num_bbox, 4]
            cls_batch: [num_bbox]
            '''
            coord_batch = np.array(annot["bbox_list"])
            cls_batch = np.array(annot["lbl_list"])

            for cls in self.cls_list:
                cls_spec_num_gt_list[cls] += np.sum(cls_batch == cls)

            annot_pred = self.detect(img, conf_score_thre, iou_thre)

            '''
            coord_pred_batch: [num_bbox_pred, 4]
            cls_pred_batch: [num_bbox_pred]
            cls_spec_conf_score_pred_batch: [num_bbox_pred, num_cls]
            '''
            coord_pred_batch = np.array(annot_pred["bbox_list"])
            cls_pred_batch = np.array(annot_pred["lbl_list"])
            cls_spec_conf_score_pred_batch = np.array(
                annot_pred["cls_spec_conf_score_list"]
            )

            for coord_pred, cls_pred, cls_spec_conf_score_pred in zip(
                coord_pred_batch,
                cls_pred_batch,
                cls_spec_conf_score_pred_batch,
            ):
                '''cls_mask_batch: [num_bbox]'''
                cls_mask_batch = cls_batch == cls_pred

                '''
                coord_tgt_batch: [num_bbox_tgt, 4]
                cls_tgt_batch: [num_bbox_tgt]
                '''
                coord_tgt_batch = coord_batch[np.where(cls_mask_batch)]

                '''coord_pred: [1, 4]'''
                coord_pred = np.array([coord_pred])

                '''iou_batch: [num_bbox_tgt]'''
                iou_batch = get_iou(
                    coord_tgt_batch, coord_pred, numpy=True
                )

                '''
                tp, fp: [num_bbox_tgt, num_level] -> [num_level]
                '''
                fp = (
                    (np.expand_dims(iou_batch, axis=-1) < level_list)
                    .astype(int)
                )
                fp = (fp.prod(0) >= 1).astype(int)

                tp = 1 - fp

                cls_spec_tp_list[cls_pred].append(tp)
                cls_spec_fp_list[cls_pred].append(fp)
                (
                    cls_spec_conf_score_list[cls_pred]
                    .append(cls_spec_conf_score_pred)
                )

        # print("=====================================")
        # print(cls_spec_tp_list)
        # print(cls_spec_fp_list)
        # print(cls_spec_conf_score_list)
        # print(cls_spec_num_gt_list)
        # print("=====================================")

        for cls in self.cls_list:
            print("=====================================")
            print(cls)
            '''
            tp_list: [num_bbox, num_level]
            fp_list: [num_bbox, num_level]
            conf_score_list: [num_bbox]
            num_gt: []
            '''
            tp_list = np.vstack(cls_spec_tp_list[cls])
            fp_list = np.vstack(cls_spec_fp_list[cls])
            conf_score_list = np.array(cls_spec_conf_score_list[cls])
            num_gt = cls_spec_num_gt_list[cls]

            sorted_indices = np.argsort(conf_score_list)[::-1]

            '''
            sorted_tp_list: [num_bbox, num_level]
            sorted_fp_list: [num_bbox, num_level]
            sorted_conf_score_list: [num_bbox]
            '''
            sorted_tp_list = tp_list[sorted_indices]
            sorted_fp_list = fp_list[sorted_indices]
            # sorted_conf_score_list = conf_score_list[sorted_indices]

            '''
            sorted_tp_list_cumsum: [num_bbox, num_level]
            sorted_fp_list_cumsum: [num_bbox, num_level]
            '''
            sorted_tp_list_cumsum = np.cumsum(sorted_tp_list, axis=0)
            sorted_fp_list_cumsum = np.cumsum(sorted_fp_list, axis=0)

            '''
            sorted_prec_list: [num_bbox, num_level]
            sorted_rec_list: [num_bbox, num_level]
            '''
            sorted_prec_list = (
                sorted_tp_list_cumsum /
                (sorted_tp_list_cumsum + sorted_fp_list_cumsum + eps)
            )
            sorted_rec_list = sorted_tp_list_cumsum / (num_gt + eps)

            # print("sorted_prec_list, sorted_rec_list")
            # print(sorted_prec_list.T)
            # print(sorted_rec_list.T)

            '''sorted_prec_list_reverse_cummax: [num_bbox, num_level]'''
            '''sorted_rec_diff_list: [num_bbox, num_level]'''
            sorted_prec_list_reverse_cummax = (
                cummax(sorted_prec_list[::-1], axis=0)[::-1]
            )
            # print("sorted_prec_list_reverse_cummax, sorted_rec_diff_list")
            # print(sorted_prec_list_reverse_cummax.T)

            sorted_rec_diff_list = np.zeros_like(sorted_rec_list)
            sorted_rec_diff_list[1:] = sorted_rec_list[:-1]
            # print(sorted_rec_diff_list.T)
            sorted_rec_diff_list = sorted_rec_list - sorted_rec_diff_list
            # print(sorted_rec_diff_list.T)

            '''ap: []'''
            ap = np.sum(
                sorted_prec_list_reverse_cummax * sorted_rec_diff_list, axis=0
            )
            print(ap)

    def collate_fn_with_imgaug(self, batch):
        return self.collate_fn(batch, augmentation=True)

    def collate_fn(self, batch, augmentation=False):
        num_cls = self.num_cls

        x_batch = []
        bbox_norm_batch = []
        bbox_coord_batch = []
        cls_tgt_batch = []
        obj_mask_batch = []
        x_img_id_batch = []
        bbox_img_id_batch = []

        for img_id, img, annot in batch:
            if augmentation:
                transformed = self.transform(
                    image=img, bboxes=annot["bbox_list"],
                    labels=annot["lbl_list"],
                )

                img = transformed["image"]
                annot = {
                    "bbox_list": transformed["bboxes"],
                    "lbl_list": transformed["labels"],
                }

                transformed = self.resize(
                    image=img, bboxes=annot["bbox_list"],
                    labels=annot["lbl_list"],
                )

                img = transformed["image"]
                annot = {
                    "bbox_list": transformed["bboxes"],
                    "lbl_list": transformed["labels"],
                }

            x = torch.tensor(img).to(DEVICE)
            x_batch.append(x)
            x_img_id_batch.append(img_id)

            height, width, _ = x.shape

            num_grid_cell_in_height = height // 32
            num_grid_cell_in_width = width // 32

            grid_cell_height = height / num_grid_cell_in_height
            grid_cell_width = width / num_grid_cell_in_width

            for bbox, lbl in zip(annot["bbox_list"], annot["lbl_list"]):
                '''
                bbox_norm: [..., [bx_norm, by_norm, bw, bh]]
                bbox_coord: [..., [x1, y1, x2, y2]]
                '''
                bbox_norm = np.zeros(
                    shape=[num_grid_cell_in_height, num_grid_cell_in_width, 4]
                )
                bbox_coord = np.zeros(
                    shape=[num_grid_cell_in_height, num_grid_cell_in_width, 4]
                )
                cls_tgt = np.zeros(
                    shape=[
                        num_grid_cell_in_height,
                        num_grid_cell_in_width,
                        num_cls
                    ]
                )
                obj_mask = np.zeros(
                    shape=[num_grid_cell_in_height, num_grid_cell_in_width]
                )

                x1, y1, x2, y2 = bbox

                x1_norm = x1 / grid_cell_width
                y1_norm = y1 / grid_cell_height
                x2_norm = x2 / grid_cell_width
                y2_norm = y2 / grid_cell_height

                bx = (x1_norm + x2_norm) / 2
                by = (y1_norm + y2_norm) / 2
                bw = x2_norm - x1_norm
                bh = y2_norm - y1_norm

                cx = int(bx)
                cy = int(by)

                bx_norm = bx - cx
                by_norm = by - cy

                bbox_norm[cy, cx, 0] = bx_norm
                bbox_norm[cy, cx, 1] = by_norm
                bbox_norm[cy, cx, 2] = bw
                bbox_norm[cy, cx, 3] = bh

                bbox_coord[cy, cx, 0] = x1
                bbox_coord[cy, cx, 1] = y1
                bbox_coord[cy, cx, 2] = x2
                bbox_coord[cy, cx, 3] = y2

                cls = lbl
                cls_idx = self.cls2idx[cls]

                cls_tgt[cy, cx, cls_idx] = 1

                obj_mask[cy, cx] = 1

                bbox_norm = torch.tensor(bbox_norm).to(DEVICE)
                bbox_coord = torch.tensor(bbox_coord).to(DEVICE)
                cls_tgt = torch.tensor(cls_tgt).to(DEVICE)
                obj_mask = torch.tensor(obj_mask).to(DEVICE)

                bbox_norm_batch.append(bbox_norm)
                bbox_coord_batch.append(bbox_coord)
                cls_tgt_batch.append(cls_tgt)
                obj_mask_batch.append(obj_mask)
                bbox_img_id_batch.append(img_id)

        '''
        x_batch:
            - [batch_size, height, width, rgb]
        bbox_norm_batch:
            - [num_bbox, num_grid_cell_in_height, num_grid_cell_in_width, 4]
        bbox_coord_batch:
            - [num_bbox, num_grid_cell_in_height, num_grid_cell_in_width, 4]
        cls_tgt_batch:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_cls,
            ]
        obj_mask_batch:
            - [num_bbox, num_grid_cell_in_height, num_grid_cell_in_width]
        x_img_id_batch:
            - [batch_size]
        bbox_img_id_batch:
            - [num_bbox]
        '''
        x_batch = torch.stack(x_batch, dim=0)
        bbox_norm_batch = torch.stack(bbox_norm_batch, dim=0)
        bbox_coord_batch = torch.stack(bbox_coord_batch, dim=0)
        cls_tgt_batch = torch.stack(cls_tgt_batch, dim=0)
        obj_mask_batch = torch.stack(obj_mask_batch, dim=0)
        x_img_id_batch = torch.tensor(x_img_id_batch).to(DEVICE)
        bbox_img_id_batch = torch.tensor(bbox_img_id_batch).to(DEVICE)

        return (
            x_batch,
            bbox_norm_batch,
            bbox_coord_batch,
            cls_tgt_batch,
            obj_mask_batch,
            x_img_id_batch,
            bbox_img_id_batch,
        )

    def get_random_size_transform(self):
        target_size_list = 32 * np.arange(10, 20)

        target_size = np.random.choice(target_size_list)

        transform = albumentations.Compose(
            [
                albumentations.Resize(target_size, target_size)
            ],
            bbox_params=albumentations.BboxParams(
                format="pascal_voc", label_fields=["labels"]
            ),
        )

        return transform
