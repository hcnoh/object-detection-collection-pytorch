import os
import pickle

import numpy as np
import torch
import torch.cuda
import torch.backends.mps
import imgaug as ia
import imgaug.augmenters as iaa

from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, LeakyReLU
from torch.nn.functional import one_hot
from torch.optim import SGD

from config import DEVICE
from models.backbones.darknet19 import Darknet19Backbone
from models.utils import get_iou, get_aps


class YOLOv2(Module):
    def __init__(
        self,
        S,
        cls_list,
        cls2idx,
    ) -> None:
        super().__init__()

        self.anchor_box_sizes = [
            (1.3221, 1.73145),
            (3.19275, 4.00944),
            (5.05587, 8.09892),
            (9.47112, 4.84053),
            (11.2364, 10.0071),
        ]
        self.B = len(self.anchor_box_sizes)

        self.pw_list = (
            torch.tensor([b[0] for b in self.anchor_box_sizes]).to(DEVICE)
        )
        self.ph_list = (
            torch.tensor([b[1] for b in self.anchor_box_sizes]).to(DEVICE)
        )

        self.S = S

        self.cls_list = cls_list
        self.cls2idx = cls2idx

        self.C = len(self.cls_list)

        self.head_output_dim = self.B * (5 + self.C)

        self.backbone_model = Darknet19Backbone()

        self.h_in = self.backbone_model.h_in
        self.w_in = self.backbone_model.w_in

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
                    - [N, H, W, C] = [N, 416, 416, 3]

            Returns:
                h1:
                    - [N, 512, 26, 26]
                h2:
                    - [N, 1024, 13, 13]
        '''
        # x: [N, C, H, W] = [N, 3, 416, 416]
        x = self.backbone_model.normalize(x)

        # h: [N, 32, 416, 416]
        h = self.backbone_model.net1(x)

        # h: [N, 64, 208, 208]
        h = self.backbone_model.net2(h)

        # h: [N, 128, 104, 104]
        h = self.backbone_model.net3(h)

        # h: [N, 256, 52, 52]
        h = self.backbone_model.net4(h)

        # h1: [N, 512, 26, 26]
        h1 = self.backbone_model.net5(h)

        # h2: [N, 1024, 13, 13]
        h2 = self.backbone_model.net6(h1)

        # h2: [N, 1024, 13, 13]
        h2 = self.backbone_model.net7(h2)

        return h1, h2

    def neck(self, h1, h2):
        '''
            Args:
                h1:
                    - [N, 512, 26, 26]
                h2:
                    - [N, 1024, 13, 13]

            Returns:
                h:
                    - [N, 3072, 13, 13]
        '''
        N, _, _, _ = h1.shape

        # h1: [N, 26, 26, 512] -> [N, 26, 13, 1024] -> [N, 2048, 13, 13]
        h1 = h1.permute(0, 2, 3, 1)
        h1 = h1.reshape([N, 26, 13, 1024])
        h1 = (
            h1.permute(0, 2, 1, 3)
            .reshape([N, 13, 13, 2048]).permute(0, 3, 2, 1)
        )

        # h: [N, 3072, 13, 13]
        h = torch.cat([h1, h2], dim=1)

        return h

    def head(self, h):
        '''
            Args:
                h:
                    - [N, 3072, 13, 13]

            Returns:
                y:
                    - [N, 13, 13, output_dim]
        '''
        B = self.B
        C = self.C

        # pw, ph: [1, 1, 1, B]
        pw = self.pw_list.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        ph = self.ph_list.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # y: [N, S, S, B * (5 + C)]
        y = self.head_model(h).permute(0, 2, 3, 1)

        # tx, ty -> sigmoid(tx), sigmoid(ty)
        y[..., 0::5 + C] = torch.sigmoid(y[..., 0::5 + C])
        y[..., 1::5 + C] = torch.sigmoid(y[..., 1::5 + C])

        # tw, th -> pw * exp(tw), ph * exp(th)
        y[..., 2::5 + C] = pw * torch.exp(y[..., 2::5 + C])
        y[..., 3::5 + C] = ph * torch.exp(y[..., 3::5 + C])

        # to -> sigmoid(to)
        y[..., 4::5 + C] = torch.sigmoid(y[..., 4::5 + C])

        # cond_cls_prob
        for i in range(B):
            y[..., i * (C + 5) + 5:i * (C + 5) + 5 + C] = (
                torch.softmax(
                    y[..., i * (C + 5) + 5:i * (C + 5) + 5 + C],
                    dim=-1
                )
            )

        # for i in range(5, C + 5):
        #     y[..., i::5 + C] = torch.sigmoid(y[..., i::5 + C])

        return y

    def forward(self, x):
        '''
            Args:
                x:
                    - the input image whose type is FloatTensor
                    - [N, H, W, C] = [N, 416, 416, 3]

            Returns:
                y:
                    - [N, 13, 13, output_dim]
        '''
        # h1: [N, 512, 26, 26]
        # h2: [N, 1024, 13, 13]
        h1, h2 = self.backbone(x)

        # h: [N, 3072, 13, 13]
        h = self.neck(h1, h2)

        # y: [N, S, S, B * (5 + C)]
        y = self.head(h)

        return y

    def get_loss(
        self,
        y_pred_batch,
        y_tgt_batch,
        coord_batch,
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
            N: batch_size
            M: the # of bounding boxes in the given batch

            Args:
                y_pred_batch:
                    - the model's prediction on the given targets
                    - [N, S, S, B * 5 + C]
                y_tgt_batch:
                    - the given targets
                    - [M, S, S, 4]
                coord_batch:
                    - the given coordinates for the all bounding boxes
                    - [M, S, S, 4]
                cls_tgt_batch:
                    - the given class targets as onehot vectors
                    - [M, S, S, C]
                obj_mask_batch:
                    - the mask to indicate the object is in each grid cell
                    - [M, S, S]
                x_img_id_batch:
                    - the image IDs for the given input
                    - [N]
                bbox_img_id_batch:
                    - the image IDs for the given bounding boxes
                    - [M]
        '''
        eps = 1e-6

        S = self.S
        B = self.B
        C = self.C

        # pw, ph: [1, 1, 1, B]
        pw = self.pw_list.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        ph = self.ph_list.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # bbox_img_id_to_x_img_id_mapper: [M, N] -> [M]
        bbox_img_id_to_x_img_id_mapper = (
            (
                bbox_img_id_batch.unsqueeze(-1) ==
                x_img_id_batch.unsqueeze(0)
            ).long()
            .argmax(-1)
        )

        # y_pred_batch: [N, S, S, B * (5 + C)] -> [M, S, S, B * (5 + C)]
        y_pred_batch = y_pred_batch[bbox_img_id_to_x_img_id_mapper]

        # bx_norm_pred_batch, by_norm_pred_batch : [M, S, S, B]
        # bw_pred_batch, bh_pred_batch: [M, S, S, B]
        # conf_score_pred_batch: [M, S, S, B]
        # cond_cls_prob_pred_batch: [M, S, S, B, C]
        bx_norm_pred_batch = y_pred_batch[..., 0::5 + C]
        by_norm_pred_batch = y_pred_batch[..., 1::5 + C]
        bw_pred_batch = y_pred_batch[..., 2::5 + C]
        bh_pred_batch = y_pred_batch[..., 3::5 + C]
        conf_score_pred_batch = y_pred_batch[..., 4::5 + C]
        cond_cls_prob_pred_batch = torch.stack(
            [y_pred_batch[..., i::5 + C] for i in range(5, 5 + C)],
            dim=-1
        )

        # tx_pred_batch, ty_pred_batch,
        # tw_pred_batch, th_pred_batch: [M, S, S, B]
        tx_pred_batch = torch.logit(bx_norm_pred_batch, eps)
        ty_pred_batch = torch.logit(by_norm_pred_batch, eps)
        tw_pred_batch = torch.log(bw_pred_batch / pw + eps)
        th_pred_batch = torch.log(bh_pred_batch / ph + eps)

        # bx_norm_tgt_batch, by_norm_tgt_batch: [M, S, S, 1]
        # bw_tgt_batch, bh_tgt_batch: [M, S, S, 1]
        bx_norm_tgt_batch = y_tgt_batch[..., 0].unsqueeze(-1).clamp(0., 1.)
        by_norm_tgt_batch = y_tgt_batch[..., 1].unsqueeze(-1).clamp(0., 1.)
        bw_tgt_batch = y_tgt_batch[..., 2].unsqueeze(-1)
        bh_tgt_batch = y_tgt_batch[..., 3].unsqueeze(-1)

        # tx_tgt_batch, ty_tgt_batch,
        # tw_tgt_batch, th_tgt_batch: [M, S, S, B]
        tx_tgt_batch = torch.logit(bx_norm_tgt_batch, eps)
        ty_tgt_batch = torch.logit(by_norm_tgt_batch, eps)
        tw_tgt_batch = torch.log(bw_tgt_batch / pw + eps)
        th_tgt_batch = torch.log(bh_tgt_batch / ph + eps)

        # cls_tgt_batch: [M, S, S, 1, C]
        cls_tgt_batch = cls_tgt_batch.unsqueeze(-2)

        # obj_mask_batch: [M, S, S, 1]
        obj_mask_batch = obj_mask_batch.unsqueeze(-1)

        # cy_batch: [1, S, 1, 1]
        # cx_batch: [1, 1, S, 1]
        cy_batch = (
            torch.arange(S).to(DEVICE)
            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        cx_batch = (
            torch.arange(S).to(DEVICE)
            .unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )

        # bx_pred_batch, by_pred_batch: [M, S, S, B]
        bx_pred_batch = (
            cx_batch + bx_norm_pred_batch
        )
        by_pred_batch = (
            cy_batch + by_norm_pred_batch
        )

        # x1_pred_batch, x2_pred_batch,
        # y1_pred_batch, y2_pred_batch: [M, S, S, B]
        x1_pred_batch = bx_pred_batch - (bw_pred_batch / 2)
        x2_pred_batch = bx_pred_batch + (bw_pred_batch / 2)
        y1_pred_batch = by_pred_batch - (bh_pred_batch / 2)
        y2_pred_batch = by_pred_batch + (bh_pred_batch / 2)

        # x1_tgt_batch, x2_tgt_batch,
        # y1_tgt_batch, y2_tgt_batch: [M, S, S, 1]
        x1_tgt_batch = coord_batch[..., 0].unsqueeze(-1)
        x2_tgt_batch = coord_batch[..., 1].unsqueeze(-1)
        y1_tgt_batch = coord_batch[..., 2].unsqueeze(-1)
        y2_tgt_batch = coord_batch[..., 3].unsqueeze(-1)

        # iou_batch: [M, S, S, B]
        iou_batch = get_iou(
            x1_tgt_batch,
            x2_tgt_batch,
            y1_tgt_batch,
            y2_tgt_batch,
            x1_pred_batch,
            x2_pred_batch,
            y1_pred_batch,
            y2_pred_batch,
        ).detach()

        # responsible_mask_batch: [M, S, S] -> [M, S, S, B]
        # not_responsible_mask_batch: [M, S, S, B]
        _, responsible_mask_batch = (
            torch.max(iou_batch, dim=-1)
        )
        responsible_mask_batch = one_hot(responsible_mask_batch, B)
        responsible_mask_batch = (responsible_mask_batch * obj_mask_batch)

        not_responsible_mask_batch = (responsible_mask_batch != 1)

        responsible_mask_batch = responsible_mask_batch.bool()
        not_responsible_mask_batch = not_responsible_mask_batch.bool()

        # loss_xy: [M, S, S, B] -> [M] -> []
        loss_xy = (
            (tx_tgt_batch - tx_pred_batch) ** 2 +
            (ty_tgt_batch - ty_pred_batch) ** 2
        )
        # loss_xy = (loss_xy * responsible_mask_batch).sum(-1).sum(-1).sum(-1)
        loss_xy = torch.masked_select(loss_xy, responsible_mask_batch)
        loss_xy = loss_xy.mean()

        # loss_wh: [M, S, S, B] -> [M] -> []
        # loss_wh = (
        #     (torch.sqrt(bw_tgt_batch) - torch.sqrt(bw_pred_batch)) ** 2 +
        #     (torch.sqrt(bh_tgt_batch) - torch.sqrt(bh_pred_batch)) ** 2
        # )
        loss_wh = (
            (tw_tgt_batch - tw_pred_batch) ** 2 +
            (th_tgt_batch - th_pred_batch) ** 2
        )
        # loss_wh = (loss_wh * responsible_mask_batch).sum(-1).sum(-1).sum(-1)
        loss_wh = torch.masked_select(loss_wh, responsible_mask_batch)
        loss_wh = loss_wh.mean()

        # loss_conf: [M, S, S, B] -> [M] -> []
        loss_conf = (1 - conf_score_pred_batch) ** 2
        # loss_conf = (
        #     (loss_conf * responsible_mask_batch)
        #     .sum(-1).sum(-1).sum(-1)
        # )
        loss_conf = torch.masked_select(loss_conf, responsible_mask_batch)
        loss_conf = loss_conf.mean()

        # loss_noobj: [M, S, S, B] -> [M * S * S * B - M] -> []
        loss_noobj = (0 - conf_score_pred_batch) ** 2
        # loss_noobj = (
        #     (loss_noobj * not_responsible_mask_batch)
        #     .sum(-1).sum(-1).sum(-1)
        # )
        loss_noobj = torch.masked_select(
            loss_noobj, not_responsible_mask_batch
        )
        loss_noobj = loss_noobj.mean()

        # loss_cls: [M, S, S, B, C] -> [M, S, S, B] -> [M] -> []
        loss_cls = (cls_tgt_batch - cond_cls_prob_pred_batch) ** 2
        # loss_cls = (
        #     (loss_cls * responsible_mask_batch.unsqueeze(-1))
        #     .sum(-1).sum(-1).sum(-1).sum(-1)
        # )
        loss_cls = loss_cls.sum(-1)
        loss_cls = torch.masked_select(loss_cls, responsible_mask_batch)
        loss_cls = loss_cls.mean()

        # loss: []
        loss = (
            lambda_xy * loss_xy +
            lambda_wh * loss_wh +
            lambda_conf * loss_conf +
            lambda_noobj * loss_noobj +
            lambda_cls * loss_cls
        )
        # loss = loss.mean()

        return loss

    def execute_one_step(
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
        # iou_batch = []
        # cls_tgt_batch = []
        # cls_score_batch = []
        # bbox_img_id_batch = []

        dataset_size = len(data_loader.dataset)
        progress_size = 0

        for batch in data_loader:
            # N: batch_size
            # M: the # of bboxes in the given batch

            # x_batch_one_step: [N, H, W, 3]
            # y_tgt_batch_one_step: [M, S, S, 4]
            # coord_batch_one_step: [M, S, S, 4]
            # cls_tgt_batch_one_step: [M, S, S, C]
            # obj_mask_batch_one_step: [M, S, S]
            # x_img_id_batch_one_step: [N]
            # bbox_img_id_batch_one_step: [M]
            (
                x_batch_one_step,
                y_tgt_batch_one_step,
                coord_batch_one_step,
                cls_tgt_batch_one_step,
                obj_mask_batch_one_step,
                x_img_id_batch_one_step,
                bbox_img_id_batch_one_step,
            ) = batch

            N = x_batch_one_step.shape[0]

            progress_size += N

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

            # y_pred_batch_one_step: [N, S, S, B * 5 + C]
            y_pred_batch_one_step = self(x_batch_one_step)

            # loss_one_step: []
            loss_one_step = self.get_loss(
                y_pred_batch_one_step,
                y_tgt_batch_one_step,
                coord_batch_one_step,
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
        num_epochs_list,
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

        for lr, num_epochs in zip(learning_rate_list, num_epochs_list):

            for epoch in range(1 + cum_epoch, num_epochs + 1 + cum_epoch):
                train_loss_mean = self.execute_one_step(
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
                val_loss = self.execute_one_step(
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

    def evaluate_one_step(
        self,
        y_pred_batch,
        coord_batch,
        cls_tgt_batch,
        x_img_id_batch,
        bbox_img_id_batch,
    ):
        '''
            N: batch_size
            M: the # of bounding boxes in the given batch

            Args:
                y_pred_batch:
                    - the model's prediction on the given targets
                    - [N, S, S, B * 5 + C]
                coord_batch:
                    - the given coordinates for the all bounding boxes
                    - [M, S, S, 4]
                cls_tgt_batch:
                    - the given class targets as onehot vectors
                    - [M, S, S, C]
                x_img_id_batch:
                    - the image IDs for the given input
                    - [N]
                bbox_img_id_batch:
                    - the image IDs for the given bounding boxes
                    - [M]
        '''
        S = self.S
        C = self.C

        # bbox_img_id_to_x_img_id_mapper: [M, N] -> [M]
        bbox_img_id_to_x_img_id_mapper = (
            (
                bbox_img_id_batch.unsqueeze(-1) ==
                x_img_id_batch.unsqueeze(0)
            ).long()
            .argmax(-1)
        )

        # y_pred_batch: [N, S, S, B * (5 + C)] -> [M, S, S, B * (5 + C)]
        y_pred_batch = y_pred_batch[bbox_img_id_to_x_img_id_mapper]

        # bx_norm_pred_batch, by_norm_pred_batch : [M, S, S, B]
        # bw_pred_batch, bh_pred_batch: [M, S, S, B]
        # conf_score_pred_batch: [M, S, S, B]
        # cond_cls_prob_pred_batch: [M, S, S, B, C]
        bx_norm_pred_batch = y_pred_batch[..., 0::5 + C]
        by_norm_pred_batch = y_pred_batch[..., 1::5 + C]
        bw_pred_batch = y_pred_batch[..., 2::5 + C]
        bh_pred_batch = y_pred_batch[..., 3::5 + C]
        conf_score_pred_batch = y_pred_batch[..., 4::5 + C]
        cond_cls_prob_pred_batch = torch.stack(
            [y_pred_batch[..., i::5 + C] for i in range(5, 5 + C)],
            dim=-1
        )

        # cy_batch: [1, S, 1, 1]
        # cx_batch: [1, 1, S, 1]
        cy_batch = (
            torch.arange(S).to(DEVICE)
            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        cx_batch = (
            torch.arange(S).to(DEVICE)
            .unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )

        # bx_pred_batch, by_pred_batch: [M, S, S, B]
        bx_pred_batch = (
            cx_batch + bx_norm_pred_batch
        )
        by_pred_batch = (
            cy_batch + by_norm_pred_batch
        )

        # x1_pred_batch, x2_pred_batch,
        # y1_pred_batch, y2_pred_batch: [M, S, S, B]
        x1_pred_batch = bx_pred_batch - (bw_pred_batch / 2)
        x2_pred_batch = bx_pred_batch + (bw_pred_batch / 2)
        y1_pred_batch = by_pred_batch - (bh_pred_batch / 2)
        y2_pred_batch = by_pred_batch + (bh_pred_batch / 2)

        # x1_tgt_batch, x2_tgt_batch,
        # y1_tgt_batch, y2_tgt_batch: [M, S, S, 1]
        x1_tgt_batch = coord_batch[:, :, :, 0].unsqueeze(-1)
        x2_tgt_batch = coord_batch[:, :, :, 1].unsqueeze(-1)
        y1_tgt_batch = coord_batch[:, :, :, 2].unsqueeze(-1)
        y2_tgt_batch = coord_batch[:, :, :, 3].unsqueeze(-1)

        # iou_batch: [M, S, S, B]
        iou_batch = get_iou(
            x1_tgt_batch,
            x2_tgt_batch,
            y1_tgt_batch,
            y2_tgt_batch,
            x1_pred_batch,
            x2_pred_batch,
            y1_pred_batch,
            y2_pred_batch,
        ).detach()

        # cls_tgt_batch: [M, S, S, C] -> [M, C]
        cls_tgt_batch = (cls_tgt_batch.sum(1).sum(1) != 0)

        # cls_spec_conf_score_pred_batch: [M, S, S, B, C]
        cls_spec_conf_score_pred_batch = (
            cond_cls_prob_pred_batch *
            conf_score_pred_batch.unsqueeze(-1)
        )

        # iou_batch: [M, S, S, B]
        # cls_tgt_batch: [M, C]
        # cls_score_batch: [M, S, S, B, C]
        # bbox_img_id_batch: [M]
        iou_batch = iou_batch.detach().cpu().numpy()
        cls_tgt_batch = cls_tgt_batch.detach().cpu().numpy()
        cls_score_batch = cls_spec_conf_score_pred_batch.detach().cpu().numpy()
        bbox_img_id_batch = bbox_img_id_batch.detach().cpu().numpy()

        return (
            iou_batch,
            cls_tgt_batch,
            cls_score_batch,
            bbox_img_id_batch,
        )

    def evaluate_model(
        self,
        data_loader,
        ckpt_path,
    ):
        iou_batch = []
        cls_tgt_batch = []
        cls_score_batch = []
        bbox_img_id_batch = []

        dataset_size = len(data_loader.dataset)
        progress_size = 0

        for batch in data_loader:
            # N: batch_size
            # M: the # of bboxes in the given batch

            # x_batch_one_step: [N, H, W, 3]
            # coord_batch_one_step: [M, S, S, 4]
            # cls_tgt_batch_one_step: [M, S, S, C]
            # x_img_id_batch_one_step: [N]
            # bbox_img_id_batch_one_step: [M]
            (
                x_batch_one_step,
                _,
                coord_batch_one_step,
                cls_tgt_batch_one_step,
                _,
                x_img_id_batch_one_step,
                bbox_img_id_batch_one_step,
            ) = batch

            N = x_batch_one_step.shape[0]

            progress_size += N

            print(
                "Evaluation: [{} / {}]".format(progress_size, dataset_size),
                end="\r"
            )

            self.eval()

            # y_pred_batch_one_step: [N, S, S, B * 5 + C]
            y_pred_batch_one_step = self(x_batch_one_step)

            # iou_batch_one_step: [M, S, S, B]
            # cls_tgt_batch_one_step: [M, C]
            # cls_score_batch_one_step: [M, S, S, B, C]
            # bbox_img_id_batch_one_step: [M]
            (
                iou_batch_one_step,
                cls_tgt_batch_one_step,
                cls_score_batch_one_step,
                bbox_img_id_batch_one_step,
            ) = self.evaluate_one_step(
                y_pred_batch_one_step,
                coord_batch_one_step,
                cls_tgt_batch_one_step,
                x_img_id_batch_one_step,
                bbox_img_id_batch_one_step,
            )

            iou_batch.append(iou_batch_one_step)
            cls_tgt_batch.append(cls_tgt_batch_one_step)
            cls_score_batch.append(cls_score_batch_one_step)
            bbox_img_id_batch.append(bbox_img_id_batch_one_step)

            # if progress_size > 500:
            #     break

        # iou_batch: [M, S, S, B]
        # cls_tgt_batch: [M, C]
        # cls_score_batch: [M, S, S, B, C]
        # bbox_img_id_batch: [M]
        iou_batch = np.vstack(iou_batch)
        cls_tgt_batch = np.vstack(cls_tgt_batch)
        cls_score_batch = np.vstack(cls_score_batch)
        bbox_img_id_batch = np.hstack(bbox_img_id_batch)

        aps = get_aps(
            iou_batch,
            cls_tgt_batch,
            cls_score_batch,
            bbox_img_id_batch,
        )

        print("Evaluation Results:")
        print(aps)

        with open(
            os.path.join(ckpt_path, "evaluation_result.pkl"),
            "wb",
        ) as f:
            pickle.dump(aps, f)

    def collate_fn_with_imgaug(self, batch):
        return self.collate_fn(batch, augmentation=True)

    def collate_fn(self, batch, augmentation=False):
        w_in = self.w_in
        h_in = self.h_in

        S = self.S
        C = self.C

        grid_cell_w = w_in / S
        grid_cell_h = h_in / S

        augmenter = iaa.Sequential(
            [
                #
                # Apply the following augmenters to most images.
                #
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images

                # crop some of the images by 0-10% of their height/width
                iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))),

                # Apply affine transformations to some of the images
                # - scale to 80-120% of image height/width
                #   (each axis independently)
                # - translate by -20 to +20 relative to height/width (per axis)
                # - rotate by -45 to +45 degrees
                # - shear by -16 to +16 degrees
                # - order:
                #   - use nearest neighbour or bilinear interpolation (fast)
                # - mode:
                #   - use any available mode to fill newly created pixels
                #   - see API or scikit-image for which modes are available
                # - cval:
                #   - if the mode is constant, then use a random brightness
                #   for the newly created pixels (e.g. sometimes black,
                #   sometimes white)
                iaa.Sometimes(0.5, iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-45, 45),
                    shear=(-16, 16),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),

                #
                # Execute 0 to 5 of the following (less important) augmenters
                # per image. Don't execute all of them, as that would often be
                # way too strong.
                #
                iaa.SomeOf(
                    (0, 5),
                    [
                        # Convert some images into their superpixel
                        # representation, sample between 20 and 200
                        # superpixels per image, but do not replace all
                        # superpixels with their average, only some of them
                        # (p_replace).
                        iaa.Sometimes(
                            0.5,
                            iaa.Superpixels(
                                p_replace=(0, 1.0),
                                n_segments=(20, 200)
                            )
                        ),

                        # Blur each image with varying strength using
                        # gaussian blur (sigma between 0 and 3.0),
                        # average/uniform blur
                        # (kernel size between 2x2 and 7x7)
                        # median blur (kernel size between 3x3 and 11x11).
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)),
                            iaa.AverageBlur(k=(2, 7)),
                            iaa.MedianBlur(k=(3, 11)),
                        ]),

                        # Sharpen each image, overlay the result with the
                        # original image using an alpha between 0 (no
                        # sharpening) and 1
                        # (full sharpening effect).
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                        # Same as sharpen, but for an embossing effect.
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                        # Search in some images either for all edges or for
                        # directed edges. These edges are then marked in a
                        # black and white image and overlayed with the
                        # original image using an alpha of 0 to 0.7.
                        iaa.Sometimes(
                            0.5,
                            iaa.OneOf([
                                iaa.EdgeDetect(alpha=(0, 0.7)),
                                iaa.DirectedEdgeDetect(
                                    alpha=(0, 0.7), direction=(0.0, 1.0)
                                ),
                            ])
                        ),

                        # Add gaussian noise to some images.
                        # In 50% of these cases, the noise is randomly sampled
                        # per channel and pixel.
                        # In the other 50% of all cases it is sampled once per
                        # pixel (i.e. brightness change).
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                        ),

                        # Either drop randomly 1 to 10% of all pixels (i.e. set
                        # them to black) or drop them on an image with 2-5%
                        # percent of the original size, leading to large
                        # dropped rectangles.
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5),
                            iaa.CoarseDropout(
                                (0.03, 0.15), size_percent=(0.02, 0.05),
                                per_channel=0.2
                            ),
                        ]),

                        # Invert each image's channel with 5% probability.
                        # This sets each pixel value v to 255-v.
                        iaa.Invert(0.05, per_channel=True),

                        # Add a value of -10 to 10 to each pixel.
                        iaa.Add((-10, 10), per_channel=0.5),

                        # Change brightness of images (50-150% of original
                        # value).
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),

                        # Improve or worsen the contrast of images.
                        iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                        # Convert each image to grayscale and then overlay the
                        # result with the original with random alpha. I.e.
                        # remove colors with varying strengths.
                        iaa.Grayscale(alpha=(0.0, 1.0)),

                        # In some images move pixels locally around (with
                        # random strengths).
                        iaa.Sometimes(
                            0.5,
                            iaa.ElasticTransformation(
                                alpha=(0.5, 3.5), sigma=0.25
                            )
                        ),

                        # In some images distort local areas with varying
                        # strength.
                        iaa.Sometimes(
                            0.5,
                            iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                    ],
                    # do all of the above augmentations in random order
                    random_order=True
                )
            ],
            random_order=True
        )
        resize = iaa.Resize({"height": h_in, "width": w_in})

        x_batch = []
        y_tgt_batch = []
        coord_batch = []
        cls_tgt_batch = []
        obj_mask_batch = []
        x_img_id_batch = []
        bbox_img_id_batch = []

        for img_id, img, bbox_list in batch:
            img = np.array(img)

            if augmentation:
                img_aug, bbox_aug_list = augmenter(
                    image=img, bounding_boxes=bbox_list
                )

            else:
                img_aug, bbox_aug_list = img, bbox_list

            img_aug, bbox_aug_list = resize(
                image=img_aug, bounding_boxes=bbox_aug_list
            )

            bbox_aug_list = (
                bbox_aug_list
                .remove_out_of_image().clip_out_of_image()
            )

            x = torch.tensor(img_aug).to(DEVICE)
            x_batch.append(x)
            x_img_id_batch.append(img_id)

            for bbox in bbox_aug_list:
                y_tgt = np.zeros(shape=[S, S, 4])
                coord = np.zeros(shape=[S, S, 4])
                cls_tgt = np.zeros(shape=[S, S, C])
                obj_mask = np.zeros(shape=[S, S])

                x1 = bbox.x1 / grid_cell_w
                x2 = bbox.x2 / grid_cell_w
                y1 = bbox.y1 / grid_cell_h
                y2 = bbox.y2 / grid_cell_h

                bx = (x1 + x2) / 2
                by = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1

                cx = int(bx)
                cy = int(by)

                bx_norm = bx - cx
                by_norm = by - cy

                y_tgt[cy, cx, 0] = bx_norm
                y_tgt[cy, cx, 1] = by_norm
                y_tgt[cy, cx, 2] = bw
                y_tgt[cy, cx, 3] = bh

                coord[cy, cx, 0] = x1
                coord[cy, cx, 1] = x2
                coord[cy, cx, 2] = y1
                coord[cy, cx, 3] = y2

                cls = bbox.label
                cls_idx = self.cls2idx[cls]

                cls_tgt[cy, cx, cls_idx] = 1

                obj_mask[cy, cx] = 1

                y_tgt = torch.tensor(y_tgt).to(DEVICE)
                coord = torch.tensor(coord).to(DEVICE)
                cls_tgt = torch.tensor(cls_tgt).to(DEVICE)
                obj_mask = torch.tensor(obj_mask).to(DEVICE)

                y_tgt_batch.append(y_tgt)
                coord_batch.append(coord)
                cls_tgt_batch.append(cls_tgt)
                obj_mask_batch.append(obj_mask)
                bbox_img_id_batch.append(img_id)

        # N: batch_size
        # M: the # of bboxes in the given batch

        # x_batch: [N, H, W, 3]
        # y_tgt_batch: [M, S, S, 4]
        # coord_batch: [M, S, S, 4]
        # cls_tgt_batch: [M, S, S, C]
        # obj_mask_batch: [M, S, S]
        # x_img_id_batch: [N]
        # bbox_img_id_batch: [M]
        x_batch = torch.stack(x_batch, dim=0)
        y_tgt_batch = torch.stack(y_tgt_batch, dim=0)
        coord_batch = torch.stack(coord_batch, dim=0)
        cls_tgt_batch = torch.stack(cls_tgt_batch, dim=0)
        obj_mask_batch = torch.stack(obj_mask_batch, dim=0)
        x_img_id_batch = torch.tensor(x_img_id_batch).to(DEVICE)
        bbox_img_id_batch = torch.tensor(bbox_img_id_batch).to(DEVICE)

        return (
            x_batch,
            y_tgt_batch,
            coord_batch,
            cls_tgt_batch,
            obj_mask_batch,
            x_img_id_batch,
            bbox_img_id_batch,
        )
