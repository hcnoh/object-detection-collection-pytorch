import os
import pickle

import numpy as np
import torch
import torch.cuda
import torch.backends.mps
import imgaug.augmenters as iaa

from torch.nn import Module
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
            torch.tensor(b[0] for b in self.anchor_box_sizes).to(DEVICE)
        )
        self.ph_list = (
            torch.tensor(b[1] for b in self.anchor_box_sizes).to(DEVICE)
        )

        self.S = S

        self.cls_list = cls_list
        self.cls2idx = cls2idx

        self.C = len(self.cls_list)

        self.backbone_output_dim = self.B * (5 + self.C)

        self.backbone_model = Darknet19Backbone(self.backbone_output_dim)

        self.h_in = self.backbone_model.h_in
        self.w_in = self.backbone_model.w_in

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
        B = self.B
        C = self.C

        # pw, ph: [1, 1, 1, B]
        pw = self.pw_list.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        ph = self.ph_list.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # y: [N, S, S, B * (5 + C)]
        y = self.backbone_model(x).permute(0, 2, 3, 1)

        # tx, ty -> sigma(tx), sigma(ty)
        for i in range(0, 2):
            y[..., i:B * (5 + C):5 + C] = torch.sigmoid(
                y[..., i:B * (5 + C):5 + C]
            )

        # tw, th -> pw * exp(tw), ph * exp(th)
        y[..., 2:B * (5 + C):5 + C] = pw * torch.exp(
            y[..., 2:B * (5 + C):5 + C]
        )
        y[..., 3:B * (5 + C):5 + C] = ph * torch.exp(
            y[..., 3:B * (5 + C):5 + C]
        )

        # to -> sigma(to)
        y[..., 4:B * (5 + C):5 + C] = torch.sigmoid(
                y[..., 4:B * (5 + C):5 + C]
            )

        # cond_cls_prob
        for i in range(5, C + 5):
            y[..., i:B * (5 + C):5 + C] = torch.sigmoid(
                y[..., i:B * (5 + C):5 + C]
            )

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
        lambda_coord,
        lambda_noobj,
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
        S = self.S
        B = self.B
        C = self.C

        # bbox_img_id_to_x_img_id_mapper: [M, N] -> [M]
        bbox_img_id_to_x_img_id_mapper = (
            (
                bbox_img_id_batch.unsqueeze(-1) ==
                x_img_id_batch.unsqueeze(0)
            ).long()
            .argmax(-1)
        )

        # y_pred_batch: [N, S, S, B * 5 + C] -> [M, S, S, B * 5 + C]
        y_pred_batch = y_pred_batch[bbox_img_id_to_x_img_id_mapper]

        # bx_norm_pred_batch, by_norm_pred_batch : [M, S, S, B]
        # bw_pred_batch, bh_pred_batch: [M, S, S, B]
        # conf_score_pred_batch: [M, S, S, B]
        # cond_cls_prob_pred_batch: [M, S, S, B, C]
        bx_norm_pred_batch = y_pred_batch[..., 0:B * (5 + C):5 + C]
        by_norm_pred_batch = y_pred_batch[..., 1:B * (5 + C):5 + C]
        bw_pred_batch = y_pred_batch[..., 2:B * (5 + C):5 + C]
        bh_pred_batch = y_pred_batch[..., 3:B * (5 + C):5 + C]
        conf_score_pred_batch = y_pred_batch[..., 4:B * (5 + C):5 + C]
        cond_cls_prob_pred_batch = torch.stack(
            [y_pred_batch[..., i:B * (5 + C):5 + C] for i in range(C)],
            dim=-1
        )

        # bx_norm_tgt_batch, by_norm_tgt_batch: [M, S, S, 1]
        # bw_tgt_batch, bh_tgt_batch: [M, S, S, 1]
        bx_norm_tgt_batch = y_tgt_batch[..., 0].unsqueeze(-1).clamp(0., 1.)
        by_norm_tgt_batch = y_tgt_batch[..., 1].unsqueeze(-1).clamp(0., 1.)
        bw_tgt_batch = y_tgt_batch[..., 2].unsqueeze(-1).clamp(0., 1.)
        bh_tgt_batch = y_tgt_batch[..., 3].unsqueeze(-1).clamp(0., 1.)

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
        _, responsible_mask_batch = (
            torch.max(iou_batch, dim=-1)
        )
        responsible_mask_batch = one_hot(responsible_mask_batch, B)
        responsible_mask_batch = (responsible_mask_batch * obj_mask_batch)

        # not_responsible_mask_batch: [M, S, S, B]
        not_responsible_mask_batch = (responsible_mask_batch != 1)

        # loss_xy: [M, S, S, B] -> [M]
        loss_xy = (
            (bx_norm_tgt_batch - bx_norm_pred_batch) ** 2 +
            (by_norm_tgt_batch - by_norm_pred_batch) ** 2
        )
        loss_xy = (loss_xy * responsible_mask_batch).sum(-1).sum(-1).sum(-1)

        # loss_wh: [M, S, S, B] -> [M]
        loss_wh = (
            (torch.sqrt(bw_tgt_batch) - torch.sqrt(bw_pred_batch)) ** 2 +
            (torch.sqrt(bh_tgt_batch) - torch.sqrt(bh_pred_batch)) ** 2
        )
        loss_wh = (loss_wh * responsible_mask_batch).sum(-1).sum(-1).sum(-1)

        # loss_conf: [M, S, S, B] -> [M]
        loss_conf = (1 - conf_score_pred_batch) ** 2
        loss_conf = (
            (loss_conf * responsible_mask_batch)
            .sum(-1).sum(-1).sum(-1)
        )

        # loss_noobj: [M, S, S, B] -> [M]
        loss_noobj = (0 - conf_score_pred_batch) ** 2
        loss_noobj = (
            (loss_noobj * not_responsible_mask_batch)
            .sum(-1).sum(-1).sum(-1)
        )

        # loss_cls: [M, S, S, B, C] -> [M]
        loss_cls = (cls_tgt_batch - cond_cls_prob_pred_batch) ** 2
        loss_cls = (
            (loss_cls * responsible_mask_batch)
            .sum(-1).sum(-1).sum(-1).sum(-1)
        )

        # loss: [M] -> []
        loss = (
            lambda_coord * loss_xy +
            lambda_coord * loss_wh +
            loss_conf +
            lambda_noobj * loss_noobj +
            loss_cls
        )
        loss = loss.mean()

        return loss

    def execute_one_step(
        self,
        epoch,
        data_loader,
        lambda_coord,
        lambda_noobj,
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
                lambda_coord,
                lambda_noobj,
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
        lambda_coord,
        lambda_noobj,
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
                    lambda_coord,
                    lambda_noobj,
                    lr,
                    train=True,
                )
                val_loss = self.execute_one_step(
                    epoch,
                    val_loader,
                    lambda_coord,
                    lambda_noobj,
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
        B = self.B
        C = self.C

        # bbox_img_id_to_x_img_id_mapper: [M, N] -> [M]
        bbox_img_id_to_x_img_id_mapper = (
            (
                bbox_img_id_batch.unsqueeze(-1) ==
                x_img_id_batch.unsqueeze(0)
            ).long()
            .argmax(-1)
        )

        # y_pred_batch: [N, S, S, B * 5 + C] -> [M, S, S, B * 5 + C]
        y_pred_batch = y_pred_batch[bbox_img_id_to_x_img_id_mapper]

        # bx_norm_pred_batch, by_norm_pred_batch : [M, S, S, B]
        # bw_pred_batch, bh_pred_batch: [M, S, S, B]
        # conf_score_pred_batch: [M, S, S, B]
        # cond_cls_prob_pred_batch: [M, S, S, B, C]
        bx_norm_pred_batch = y_pred_batch[..., 0:B * (5 + C):5 + C]
        by_norm_pred_batch = y_pred_batch[..., 1:B * (5 + C):5 + C]
        bw_pred_batch = y_pred_batch[..., 2:B * (5 + C):5 + C]
        bh_pred_batch = y_pred_batch[..., 3:B * (5 + C):5 + C]
        conf_score_pred_batch = y_pred_batch[..., 4:B * (5 + C):5 + C]
        cond_cls_prob_pred_batch = torch.stack(
            [y_pred_batch[..., i:B * (5 + C):5 + C] for i in range(C)],
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

        augmenter = iaa.Sequential([
            iaa.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            ),
            iaa.MultiplyHueAndSaturation(mul_saturation=(0.5, 1.5))
        ])
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
