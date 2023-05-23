import os
import pickle

import numpy as np
import torch
import torch.cuda
import torch.backends.mps
import albumentations
import albumentations.pytorch

from torch.nn import Module, Sequential, Flatten, Linear, ReLU, Dropout
from torch.nn.functional import one_hot
from torch.optim import SGD

from config import DEVICE
from models.backbones.googlenet import GoogLeNetBackbone
from models.utils import get_iou, nms


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

RESIZE = albumentations.Compose(
    [
        albumentations.Resize(224, 224)
    ],
    bbox_params=albumentations.BboxParams(
        format="pascal_voc", label_fields=["labels"]
    ),
)


class YOLOv1(Module):
    def __init__(
        self,
        num_grid_cell_in_height,
        num_grid_cell_in_width,
        num_anchor_box,
        cls_list,
        cls2idx,
    ) -> None:
        super().__init__()

        self.num_grid_cell_in_height = num_grid_cell_in_height
        self.num_grid_cell_in_width = num_grid_cell_in_width
        self.num_anchor_box = num_anchor_box

        self.cls_list = cls_list
        self.cls2idx = cls2idx

        self.num_cls = len(self.cls_list)

        self.backbone_model = GoogLeNetBackbone()

        self.backbone_output_dim = np.prod(self.backbone_model.output_shape)
        # self.backbone_output_dim = self.backbone_model.output_shape[-1]
        self.head_output_dim = (
            self.num_grid_cell_in_height * self.num_grid_cell_in_width *
            (self.num_anchor_box * 5 + self.num_cls)
        )
        # self.head_output_dim = self.num_anchor_box * 5 + self.num_cls

        self.head_model = Sequential(
            Flatten(),
            Linear(self.backbone_output_dim, 4096),
            ReLU(),
            Dropout(.5),
            Linear(4096, self.head_output_dim)
        )

        self.transform = TRANSFORM
        self.resize = RESIZE

    def backbone(self, x):
        '''
            Args:
                x:
                    - the input image whose type is FloatTensor
                    - [batch_size, height, width, rgb]
                        = [batch_size, 416, 416, 3]

            Returns:
        '''

        '''
        h:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                1024,
            ]
        '''
        h = self.backbone_model(x)

        return h

    def head(self, h):
        '''
            Args:
                h:
                    - [
                        batch_size,
                        num_grid_cell_in_height,
                        num_grid_cell_in_width,
                        1024,
                    ]

            Returns:
                y:
                    - [batch_size, H, W, output_dim]
        '''
        (
            batch_size,
            num_grid_cell_in_height,
            num_grid_cell_in_width,
            _
        ) = h.shape
        num_anchor_box = self.num_anchor_box
        num_cls = self.num_cls

        '''
        y:
            - [
                batch_size,
                num_grid_cell_in_height * num_grid_cell_in_width *
                (num_anchor_box * 5 + num_cls)
            ] ->
            [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box * 5 + num_cls,
            ]
        '''
        y = self.head_model(h)
        y = y.reshape(
            [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box * 5 + num_cls,
            ]
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
        h:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                1024,
            ]
        '''
        h = self.backbone(x)

        '''
        y:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                (num_anchor_box * 5 + num_cls),
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

        num_anchor_box = self.num_anchor_box
        num_cls = self.num_cls

        '''
        y_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                (num_anchor_box * 5 + num_cls),
            ]
        '''
        y_pred_batch = self(x_batch)

        (
            batch_size,
            num_grid_cell_in_height,
            num_grid_cell_in_width,
            _,
        ) = y_pred_batch.shape

        '''
        txtytwthto_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                5,
            ]
        '''
        txtytwthto_pred_batch = (
            y_pred_batch[..., :num_anchor_box * 5]
            .reshape(
                [
                    batch_size,
                    num_grid_cell_in_height,
                    num_grid_cell_in_width,
                    num_anchor_box,
                    5,
                ]
            )
        )

        '''
        tx, ty -> sigmoid(tx), sigmoid(ty)

        tx_pred_batch, ty_pred_batch,
        sig_tx_pred_batch, sig_ty_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        '''
        tx_pred_batch = txtytwthto_pred_batch[..., 0]
        ty_pred_batch = txtytwthto_pred_batch[..., 1]
        sig_tx_pred_batch = torch.sigmoid(tx_pred_batch)
        sig_ty_pred_batch = torch.sigmoid(ty_pred_batch)

        '''
        tw, th -> sigmoid(tw), sigmoid(th)

        tw_pred_batch, th_pred_batch,
        sig_tw_pred_batch, sig_th_pred_batch
        bw_pred_batch, bh_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        '''
        tw_pred_batch = txtytwthto_pred_batch[..., 2]
        th_pred_batch = txtytwthto_pred_batch[..., 3]
        sig_tw_pred_batch = torch.sigmoid(tw_pred_batch)
        sig_th_pred_batch = torch.sigmoid(th_pred_batch)

        bw_pred_batch = num_grid_cell_in_width * sig_tw_pred_batch
        bh_pred_batch = num_grid_cell_in_height * sig_th_pred_batch

        '''
        sig_txty_pred_batch, exp_twth_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                2,
            ]
        '''
        sig_txty_pred_batch = torch.stack(
            [sig_tx_pred_batch, sig_ty_pred_batch], dim=-1,
        )
        sig_twth_pred_batch = torch.stack(
            [sig_tw_pred_batch, sig_th_pred_batch], dim=-1,
        )

        '''
        cy_batch:
            - [1, num_grid_cell_in_height, 1, 1]
        cx_batch:
            - [1, 1, num_grid_cell_in_width, 1]
        '''
        cy_batch = (
            torch.arange(num_grid_cell_in_height).to(DEVICE)
            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            .detach()
        )
        cx_batch = (
            torch.arange(num_grid_cell_in_width).to(DEVICE)
            .unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            .detach()
        )

        '''
        bx_pred_batch, by_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        '''
        bx_pred_batch = (
            sig_tx_pred_batch + cx_batch
        )
        by_pred_batch = (
            sig_ty_pred_batch + cy_batch
        )

        '''
        x1_pred_batch, y1_pred_batch,
        x2_pred_batch, y2_pred_batch,
        x1_norm_pred_batch, y1_norm_pred_batch,
        x2_norm_pred_batch, y2_norm_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        bbox_coord_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                4,
            ]
        '''
        x1_norm_pred_batch = bx_pred_batch - (bw_pred_batch / 2)
        y1_norm_pred_batch = by_pred_batch - (bh_pred_batch / 2)
        x2_norm_pred_batch = bx_pred_batch + (bw_pred_batch / 2)
        y2_norm_pred_batch = by_pred_batch + (bh_pred_batch / 2)

        grid_cell_height = height / num_grid_cell_in_height
        grid_cell_width = width / num_grid_cell_in_width

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
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        cond_cls_prob_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_cls,
            ]
        '''
        conf_score_pred_batch = torch.sigmoid(txtytwthto_pred_batch[..., 4])
        cond_cls_prob_pred_batch = torch.softmax(
            y_pred_batch[..., -num_cls:], dim=-1
        )

        '''
        cls_spec_conf_score_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                num_cls,
            ]
        '''
        cls_spec_conf_score_pred_batch = (
            cond_cls_prob_pred_batch.unsqueeze(-2) *
            conf_score_pred_batch.unsqueeze(-1)
        )

        return (
            sig_txty_pred_batch,
            sig_twth_pred_batch,
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

        height, width, _ = img.shape

        original_resize = self.get_resize_transform(height, width)

        transformed = self.resize(
            image=img, bboxes=[], labels=[],
        )
        trans_img = transformed["image"]

        '''x_batch: [batch_size, height, width, rgb]'''
        x_batch = torch.tensor(np.array([trans_img])).to(DEVICE)

        '''
        bbox_coord_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                4,
            ]
        conf_score_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        cls_spec_conf_score_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                num_cls,
            ]
        '''
        (
            _,
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
        bbox_coord_pred_batch[..., 0::2] = np.clip(
            bbox_coord_pred_batch[..., 0::2], a_min=0.0, a_max=224 - 1.0
        )
        bbox_coord_pred_batch[..., 1::2] = np.clip(
            bbox_coord_pred_batch[..., 1::2], a_min=0.0, a_max=224 - 1.0
        )
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

        transformed = original_resize(
            image=trans_img,
            bboxes=bbox_coord_pred_batch.tolist(),
            labels=[
                self.cls_list[cls_idx]
                for cls_idx in argmax_cls_spec_conf_score_pred_batch
            ],
        )

        annot_pred = {
            "bbox_list": transformed["bboxes"],
            "lbl_list": transformed["labels"],
            "conf_score_list": conf_score_pred_batch.tolist(),
            "cls_spec_conf_score_list": (
                max_cls_spec_conf_score_pred_batch.tolist()
            ),
        }

        return annot_pred

    def get_loss(
        self,
        x_batch,
        sig_txty_tgt_batch,
        sig_twth_tgt_batch,
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
                sig_txty_tgt_batch:
                    - [
                        num_bbox,
                        num_grid_cell_in_height,
                        num_grid_cell_in_width,
                        2,
                    ]
                sig_twth_tgt_batch:
                    - [
                        num_bbox,
                        num_grid_cell_in_height,
                        num_grid_cell_in_width,
                        2,
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
        # eps = 1e-6

        num_anchor_box = self.num_anchor_box

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
        sig_txty_pred_batch, sig_twth_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                2,
            ] ->
            [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                2,
            ]
        bbox_coord_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                4,
            ] ->
            [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                4,
            ]
        conf_score_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ] ->
            [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
            ]
        cond_cls_prob_pred_batch:
            - [
                batch_size,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_cls,
            ] ->
            [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_cls,
            ]
        cls_spec_conf_score_pred_batch:
            - [
                batch_size,
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
                num_cls,
            ]
        '''
        (
            sig_txty_pred_batch,
            sig_twth_pred_batch,
            bbox_coord_pred_batch,
            conf_score_pred_batch,
            cond_cls_prob_pred_batch,
            cls_spec_conf_score_pred_batch,
        ) = self.predict(x_batch)

        (
            sig_txty_pred_batch,
            sig_twth_pred_batch,
            bbox_coord_pred_batch,
            conf_score_pred_batch,
            cond_cls_prob_pred_batch,
            cls_spec_conf_score_pred_batch,
        ) = (
            sig_txty_pred_batch[bbox_img_id_to_x_img_id_mapper],
            sig_twth_pred_batch[bbox_img_id_to_x_img_id_mapper],
            bbox_coord_pred_batch[bbox_img_id_to_x_img_id_mapper],
            conf_score_pred_batch[bbox_img_id_to_x_img_id_mapper],
            cond_cls_prob_pred_batch[bbox_img_id_to_x_img_id_mapper],
            cls_spec_conf_score_pred_batch[bbox_img_id_to_x_img_id_mapper],
        )

        '''
        sqrt_sig_twth_pred_batch:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                2,
            ]
        '''
        sqrt_sig_twth_pred_batch = torch.sqrt(sig_twth_pred_batch)

        '''
        sig_txty_tgt_batch,
        sig_twth_tgt_batch, sqrt_sig_twth_tgt_batch:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                1,
                2,
            ]
        '''
        sig_txty_tgt_batch = sig_txty_tgt_batch.unsqueeze(-2)

        sig_twth_tgt_batch = sig_twth_tgt_batch.unsqueeze(-2)
        sqrt_sig_twth_tgt_batch = torch.sqrt(sig_twth_tgt_batch)

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
        responsible_mask_batch = (
            responsible_mask_batch * obj_mask_batch.unsqueeze(-1)
        )

        not_responsible_mask_batch = (responsible_mask_batch != 1)

        obj_mask_batch = obj_mask_batch.bool()
        responsible_mask_batch = responsible_mask_batch.bool()
        not_responsible_mask_batch = not_responsible_mask_batch.bool()

        mse_loss = torch.nn.MSELoss(reduction="none")

        '''
        loss_xy:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                2,
            ] ->
            [num_bbox] -> []
        '''
        loss_xy = mse_loss(sig_txty_tgt_batch, sig_txty_pred_batch)
        loss_xy = torch.masked_select(
            loss_xy, responsible_mask_batch.unsqueeze(-1)
        )
        loss_xy = loss_xy.mean()

        '''
        loss_wh:
            - [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
                num_anchor_box,
                2,
            ] ->
            [num_bbox] -> []
        '''
        loss_wh = mse_loss(sqrt_sig_twth_tgt_batch, sqrt_sig_twth_pred_batch)
        loss_wh = torch.masked_select(
            loss_wh, responsible_mask_batch.unsqueeze(-1)
        )
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
        loss_conf = mse_loss(iou_batch, conf_score_pred_batch)
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
        loss_noobj = conf_score_pred_batch ** 2
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
                num_cls,
            ] ->
            [
                num_bbox,
                num_grid_cell_in_height,
                num_grid_cell_in_width,
            ] ->
            [num_bbox] -> []
        '''
        loss_cls = mse_loss(cls_tgt_batch, cond_cls_prob_pred_batch)
        loss_cls = loss_cls.sum(-1)
        loss_cls = torch.masked_select(loss_cls, obj_mask_batch)
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
                sig_txty_tgt_batch_one_step,
                sig_twth_tgt_batch_one_step,
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
                    "Epoch: {} --> Training: [{} / {}]          "
                    .format(epoch, progress_size, dataset_size),
                    end="\r"
                )

                self.train()

            else:
                print(
                    "Epoch: {} --> Validation: [{} / {}]          "
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
                sig_txty_tgt_batch_one_step,
                sig_twth_tgt_batch_one_step,
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

        for lr, num_epochs in zip(learning_rate_list, num_epoch_list):

            for epoch in range(1 + cum_epoch, num_epochs + 1 + cum_epoch):
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

    def collate_fn_with_imgaug(self, batch):
        return self.collate_fn(batch, augmentation=True)

    def collate_fn(self, batch, augmentation=False):
        num_cls = self.num_cls

        x_batch = []
        sig_txty_batch = []
        sig_twth_batch = []
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

            else:
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
                sig_txty: [..., [sig_tx, sig_ty]]
                bwbh: [..., [bw, bh]]
                bbox_coord: [..., [x1, y1, x2, y2]]
                '''
                sig_txty = np.zeros(
                    shape=[num_grid_cell_in_height, num_grid_cell_in_width, 2]
                )
                sig_twth = np.zeros(
                    shape=[num_grid_cell_in_height, num_grid_cell_in_width, 2]
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

                sig_tx = bx - cx
                sig_ty = by - cy
                sig_tw = bw / num_grid_cell_in_width
                sig_th = bh / num_grid_cell_in_height

                sig_txty[cy, cx, 0] = sig_tx
                sig_txty[cy, cx, 1] = sig_ty
                sig_twth[cy, cx, 0] = sig_tw
                sig_twth[cy, cx, 1] = sig_th

                bbox_coord[cy, cx, 0] = x1
                bbox_coord[cy, cx, 1] = y1
                bbox_coord[cy, cx, 2] = x2
                bbox_coord[cy, cx, 3] = y2

                cls = lbl
                cls_idx = self.cls2idx[cls]

                cls_tgt[cy, cx, cls_idx] = 1

                obj_mask[cy, cx] = 1

                sig_txty = torch.tensor(sig_txty).to(DEVICE).float()
                sig_twth = torch.tensor(sig_twth).to(DEVICE).float()
                bbox_coord = torch.tensor(bbox_coord).to(DEVICE).float()
                cls_tgt = torch.tensor(cls_tgt).to(DEVICE).float()
                obj_mask = torch.tensor(obj_mask).to(DEVICE)

                sig_txty_batch.append(sig_txty)
                sig_twth_batch.append(sig_twth)
                bbox_coord_batch.append(bbox_coord)
                cls_tgt_batch.append(cls_tgt)
                obj_mask_batch.append(obj_mask)
                bbox_img_id_batch.append(img_id)

        '''
        x_batch:
            - [batch_size, height, width, rgb]
        sig_txty_batch:
            - [num_bbox, num_grid_cell_in_height, num_grid_cell_in_width, 2]
        sig_twth_batch:
            - [num_bbox, num_grid_cell_in_height, num_grid_cell_in_width, 2]
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
        sig_txty_batch = torch.stack(sig_txty_batch, dim=0)
        sig_twth_batch = torch.stack(sig_twth_batch, dim=0)
        bbox_coord_batch = torch.stack(bbox_coord_batch, dim=0)
        cls_tgt_batch = torch.stack(cls_tgt_batch, dim=0)
        obj_mask_batch = torch.stack(obj_mask_batch, dim=0)
        x_img_id_batch = torch.tensor(x_img_id_batch).to(DEVICE)
        bbox_img_id_batch = torch.tensor(bbox_img_id_batch).to(DEVICE)

        return (
            x_batch,
            sig_txty_batch,
            sig_twth_batch,
            bbox_coord_batch,
            cls_tgt_batch,
            obj_mask_batch,
            x_img_id_batch,
            bbox_img_id_batch,
        )

    def get_resize_transform(self, height, width):
        transform = albumentations.Compose(
            [
                albumentations.Resize(height, width)
            ],
            bbox_params=albumentations.BboxParams(
                format="pascal_voc", label_fields=["labels"]
            ),
        )

        return transform
