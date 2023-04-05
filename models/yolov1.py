import os
import pickle

import numpy as np
import torch
import torch.cuda
import torch.backends.mps
import imgaug.augmenters as iaa

from torch.nn import Module, Sequential, Flatten, Linear, ReLU, Dropout
from torch.nn.functional import one_hot
from torch.optim import SGD
from torch.nn.utils.rnn import pad_sequence

from config import DEVICE
from models.backbones.googlenet import GoogLeNetBackbone
from models.utils import get_iou, get_aps


class YOLOv1(Module):
    def __init__(
        self,
        S,
        B,
        cls_list,
        cls2idx,
    ) -> None:
        super().__init__()

        self.backbone_model = GoogLeNetBackbone()

        self.h_in = self.backbone_model.h_in
        self.w_in = self.backbone_model.w_in

        self.S = S
        self.B = B

        self.cls_list = cls_list
        self.cls2idx = cls2idx

        self.C = len(self.cls_list)

        self.backbone_output_dim = np.prod(self.backbone_model.output_shape)

        self.linear_layers = Sequential(
            Flatten(),
            Linear(self.backbone_output_dim, 4096),
            ReLU(),
            Dropout(.5),
            Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )

    def forward(self, x):
        h = self.backbone_model(x)
        h = self.linear_layers(h)
        h = h.reshape([-1, self.S, self.S, self.B * 5 + self.C])
        y = torch.sigmoid(h)

        return y

    def get_loss(
        self,
        N,
        L,
        y_pred_batch,
        tgt_batch,
        cls_idx_batch,
        mask_batch,
        lambda_coord,
        lambda_noobj,
    ):
        '''
            Args:
                N:
                    - batch size
                L:
                    - the # of bounding boxes in each target
                y_pred_batch:
                    - the model's prediction on the given targets
                    - [N, S, S, B * 5 + C]
                tgt_batch:
                    - the given targets
                    - [N, L, 12]
                cls_idx_batch:
                    - the class indices of given bounding boxes of targets
                    - [N, L]
                mask_batch:
                    - the mask for the values which is not padded value
                    - [N, L]
        '''
        w_in = self.w_in
        h_in = self.h_in

        S = self.S
        B = self.B
        C = self.C

        # bx_norm_pred_batch, by_norm_pred_batch,
        # bw_norm_pred_batch, bh_norm_pred_batch: [N, S, S, B]
        # conf_score_pred_batch: [N, S, S, B]
        # cond_cls_prob_pred_batch: [N, S, S, C]
        bx_norm_pred_batch = y_pred_batch[:, :, :, 0:B * 5:5]
        by_norm_pred_batch = y_pred_batch[:, :, :, 1:B * 5:5]
        bw_norm_pred_batch = y_pred_batch[:, :, :, 2:B * 5:5]
        bh_norm_pred_batch = y_pred_batch[:, :, :, 3:B * 5:5]
        conf_score_pred_batch = y_pred_batch[:, :, :, 4:B * 5:5]
        cond_cls_prob_pred_batch = y_pred_batch[:, :, :, -C:]

        # i_arr: [1, S, 1]
        # j_arr: [1, 1, S]
        i_arr = (
            torch.arange(S).to(DEVICE)
            .unsqueeze(0).unsqueeze(-1)
        )
        j_arr = (
            torch.arange(S).to(DEVICE)
            .unsqueeze(0).unsqueeze(0)
        )

        # i_lower_bound_arr, i_upper_bound_arr: [1, S, 1]
        # j_lower_bound_arr, j_upper_bound_arr: [1, 1, S]
        i_lower_bound_arr = i_arr * (h_in / S)
        i_upper_bound_arr = i_lower_bound_arr + (h_in / S)
        j_lower_bound_arr = j_arr * (w_in / S)
        j_upper_bound_arr = j_lower_bound_arr + (w_in / S)

        # bx_tgt_batch, by_tgt_batch: [N, L, 1, 1]
        bx_tgt_batch = tgt_batch[:, :, 4].unsqueeze(-1).unsqueeze(-1)
        by_tgt_batch = tgt_batch[:, :, 5].unsqueeze(-1).unsqueeze(-1)

        # obj_grid_cell_mask_batch: [N, L, S, S]
        obj_grid_cell_mask_batch = (
            (bx_tgt_batch >= j_lower_bound_arr.unsqueeze(1)) *
            (bx_tgt_batch < j_upper_bound_arr.unsqueeze(1)) *
            (by_tgt_batch >= i_lower_bound_arr.unsqueeze(1)) *
            (by_tgt_batch < i_upper_bound_arr.unsqueeze(1))
        )

        # noobj_grid_cell_mask_batch: [N, L, S, S]
        noobj_grid_cell_mask_batch = (
            obj_grid_cell_mask_batch == 0
        )

        # bx_pred_batch, by_pred_batch,
        # bw_pred_batch, bh_pred_batch: [N, S, S, B]
        bx_pred_batch = (
            bx_norm_pred_batch * (w_in / S) +
            j_arr.unsqueeze(-1) * (w_in / S)
        )
        by_pred_batch = (
            by_norm_pred_batch * (h_in / S) +
            i_arr.unsqueeze(-1) * (h_in / S)
        )
        bw_pred_batch = bw_norm_pred_batch * w_in
        bh_pred_batch = bh_norm_pred_batch * h_in

        # x1_pred_batch, x2_pred_batch,
        # y1_pred_batch, y2_pred_batch: [N, S, S, B]
        x1_pred_batch = bx_pred_batch - (bw_pred_batch / 2)
        x2_pred_batch = bx_pred_batch + (bw_pred_batch / 2)
        y1_pred_batch = by_pred_batch - (bh_pred_batch / 2)
        y2_pred_batch = by_pred_batch + (bh_pred_batch / 2)

        # x1_tgt_batch, x2_tgt_batch,
        # y1_tgt_batch, y2_tgt_batch: [N, L, 1, 1, 1]
        x1_tgt_batch = (
            tgt_batch[:, :, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        x2_tgt_batch = (
            tgt_batch[:, :, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        y1_tgt_batch = (
            tgt_batch[:, :, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        y2_tgt_batch = (
            tgt_batch[:, :, 3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )

        # iou_batch: [N, L, S, S, B]
        iou_batch = get_iou(
            x1_tgt_batch,
            x2_tgt_batch,
            y1_tgt_batch,
            y2_tgt_batch,
            x1_pred_batch.unsqueeze(1),
            x2_pred_batch.unsqueeze(1),
            y1_pred_batch.unsqueeze(1),
            y2_pred_batch.unsqueeze(1),
        ).detach()

        # responsible_bndbox_mask_batch: [N, L, S, S, B]
        responsible_bndbox_mask_batch = (
            iou_batch == torch.max(iou_batch, dim=-1, keepdim=True).values
        )
        responsible_bndbox_mask_batch = (
            responsible_bndbox_mask_batch *
            obj_grid_cell_mask_batch.unsqueeze(-1)
        )

        # bx_norm_tgt_batch, by_norm_tgt_batch,
        # bw_norm_tgt_batch, bh_norm_tgt_batch: [N, L]
        bx_norm_tgt_batch = tgt_batch[:, :, -4]
        by_norm_tgt_batch = tgt_batch[:, :, -3]
        bw_norm_tgt_batch = tgt_batch[:, :, -2]
        bh_norm_tgt_batch = tgt_batch[:, :, -1]

        bx_norm_tgt_batch = torch.maximum(
            torch.minimum(
                bx_norm_tgt_batch, torch.tensor(1.).to(DEVICE)
            ),
            torch.tensor(0.).to(DEVICE)
        )
        by_norm_tgt_batch = torch.maximum(
            torch.minimum(
                by_norm_tgt_batch, torch.tensor(1.).to(DEVICE)
            ),
            torch.tensor(0.).to(DEVICE)
        )
        bw_norm_tgt_batch = torch.maximum(
            torch.minimum(
                bw_norm_tgt_batch, torch.tensor(1.).to(DEVICE)
            ),
            torch.tensor(0.).to(DEVICE)
        )
        bh_norm_tgt_batch = torch.maximum(
            torch.minimum(
                bh_norm_tgt_batch, torch.tensor(1.).to(DEVICE)
            ),
            torch.tensor(0.).to(DEVICE)
        )

        # cond_cls_prob_tgt_batch: [N, L, C]
        cond_cls_prob_tgt_batch = one_hot(cls_idx_batch, C)

        # loss_xy: [N, L, S, S, B] -> [N, L]
        loss_xy = (
            (
                bx_norm_tgt_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) -
                bx_norm_pred_batch.unsqueeze(1)
            ) ** 2 +
            (
                by_norm_tgt_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) -
                by_norm_pred_batch.unsqueeze(1)
            ) ** 2
        )
        loss_xy = (
            (loss_xy * responsible_bndbox_mask_batch).sum(-1).sum(-1).sum(-1)
        )

        # loss_wh: [N, L, S, S, B] -> [N, L]
        loss_wh = (
            (
                torch.sqrt(
                    bw_norm_tgt_batch
                    .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                ) -
                torch.sqrt(
                    bw_norm_pred_batch.unsqueeze(1)
                )
            ) ** 2 +
            (
                torch.sqrt(
                    bh_norm_tgt_batch
                    .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                ) -
                torch.sqrt(
                    bh_norm_pred_batch.unsqueeze(1)
                )
            ) ** 2
        )
        loss_wh = (
            (loss_wh * responsible_bndbox_mask_batch).sum(-1).sum(-1).sum(-1)
        )

        # loss_conf: [N, L, S, S, B] -> [N, L]
        loss_conf = (
            iou_batch - conf_score_pred_batch.unsqueeze(1)
        ) ** 2
        loss_conf = (
            (loss_conf * responsible_bndbox_mask_batch)
            .sum(-1).sum(-1).sum(-1)
        )

        # loss_noobj: [N, L, S, S, B] -> [N, L]
        loss_noobj = (
            0 -
            conf_score_pred_batch.unsqueeze(1).repeat(1, L, 1, 1, 1)
        ) ** 2
        loss_noobj = (
            (loss_noobj * noobj_grid_cell_mask_batch.unsqueeze(-1))
            .sum(-1).sum(-1).sum(-1)
        )

        # loss_cls: [N, L, S, S, C] -> [N, L, S, S] -> [N, L]
        loss_cls = (
            cond_cls_prob_tgt_batch.unsqueeze(-2).unsqueeze(-2) -
            cond_cls_prob_pred_batch.unsqueeze(1)
        ) ** 2
        loss_cls = loss_cls.sum(-1)
        loss_cls = (
            (loss_cls * obj_grid_cell_mask_batch).sum(-1).sum(-1)
        )

        # loss: [N, L] -> []
        loss = (
            lambda_coord * loss_xy +
            lambda_coord * loss_wh +
            loss_conf +
            lambda_noobj * loss_noobj +
            loss_cls
        )
        loss = torch.masked_select(loss, mask=mask_batch.bool()).mean()

        # iou_list: [N, L, S, S, B] -> [L', 1]
        iou_list = torch.masked_select(
            iou_batch,
            mask=(
                (
                    mask_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                )
            ).bool()
        ).reshape([-1, 1])

        # cls_spec_conf_score_pred_batch: [N, S, S, B, C]
        cls_spec_conf_score_pred_batch = (
            cond_cls_prob_pred_batch.unsqueeze(-2) *
            conf_score_pred_batch.unsqueeze(-1)
        )

        # class_score_arr_list: [N, L, S, S, B, C] -> [L', C]
        class_score_arr_list = (
            cls_spec_conf_score_pred_batch
            .unsqueeze(1).repeat(1, L, 1, 1, 1, 1)
        )
        class_score_arr_list = torch.masked_select(
            class_score_arr_list,
            mask=(
                (
                    mask_batch
                    .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                )
            ).bool()
        ).reshape([-1, C])

        # class_true_arr_list: [N, L, S, S, B, C] -> [L', C]
        class_true_arr_list = (
            cond_cls_prob_tgt_batch
            .unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
            .repeat(1, 1, S, S, B, 1)
        )
        class_true_arr_list = torch.masked_select(
            class_true_arr_list,
            mask=(
                (
                    mask_batch
                    .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                )
            ).bool()
        ).reshape([-1, C])

        iou_list = iou_list.detach().cpu().numpy()
        class_score_arr_list = class_score_arr_list.detach().cpu().numpy()
        class_true_arr_list = class_true_arr_list.detach().cpu().numpy()

        return loss, iou_list, class_true_arr_list, class_score_arr_list

    def train_one_step(
        self,
        epoch,
        train_loader,
        lambda_coord,
        lambda_noobj,
        lr,
    ):
        train_loss_mean = []
        train_iou_list = []
        train_class_true_arr_list = []
        train_class_score_arr_list = []

        train_size = len(train_loader.dataset)
        progress_size = 0

        for data in train_loader:
            # x_batch: [N, H, W, 3]
            # tgt_batch: [N, L, 12]
            # cls_idx_batch: [N, L]
            # mask_batch: [N, L]
            x_batch, tgt_batch, cls_idx_batch, mask_batch = data

            N = x_batch.shape[0]
            L = tgt_batch.shape[1]

            progress_size += N

            print(
                "Epoch: {} --> Training: [{} / {}]"
                .format(epoch, progress_size, train_size),
                end="\r"
            )

            self.train()

            # y_pred_batch: [N, S, S, B * 5 + C]
            y_pred_batch = self(x_batch)

            (
                loss,
                iou_list,
                class_true_arr_list,
                class_score_arr_list
            ) = self.get_loss(
                N,
                L,
                y_pred_batch,
                tgt_batch,
                cls_idx_batch,
                mask_batch,
                lambda_coord,
                lambda_noobj
            )

            if epoch == 1:
                opt = SGD(
                    self.parameters(),
                    lr=lr / (10 ** (1 - (progress_size / train_size))),
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
            loss.backward()
            opt.step()

            train_loss_mean.append(loss.detach().cpu().numpy())
            train_iou_list.append(iou_list)
            train_class_true_arr_list.append(class_true_arr_list)
            train_class_score_arr_list.append(class_score_arr_list)

        train_loss_mean = np.mean(train_loss_mean)

        # iou_list: [L', 1]
        # class_true_arr_list, class_score_arr_list: [L', C]
        train_iou_list = np.vstack(train_iou_list)
        train_class_true_arr_list = np.vstack(train_class_true_arr_list)
        train_class_score_arr_list = np.vstack(train_class_score_arr_list)

        train_aps = get_aps(
            train_iou_list,
            train_class_true_arr_list,
            train_class_score_arr_list,
        )

        return train_loss_mean, train_aps

    def validate_one_step(
        self,
        epoch,
        val_loader,
        lambda_coord,
        lambda_noobj,
    ):
        val_loss_mean = []
        val_iou_list = []
        val_class_true_arr_list = []
        val_class_score_arr_list = []

        val_size = len(val_loader.dataset)
        progress_size = 0

        for data in val_loader:
            # x_batch: [N, H, W, 3]
            # tgt_batch: [N, L, 12]
            # cls_idx_batch: [N, L]
            # mask_batch: [N, L]
            x_batch, tgt_batch, cls_idx_batch, mask_batch = data

            N = x_batch.shape[0]
            L = tgt_batch.shape[1]

            progress_size += N

            print(
                "Epoch: {} --> Validation: [{} / {}]"
                .format(epoch, progress_size, val_size),
                end="\r"
            )

            self.eval()

            # y_pred_batch: [N, S, S, B * 5 + C]
            y_pred_batch = self(x_batch)

            (
                loss,
                iou_list,
                class_true_arr_list,
                class_score_arr_list
            ) = self.get_loss(
                N,
                L,
                y_pred_batch,
                tgt_batch,
                cls_idx_batch,
                mask_batch,
                lambda_coord,
                lambda_noobj
            )

            val_loss_mean.append(loss.detach().cpu().numpy())
            val_iou_list.append(iou_list)
            val_class_true_arr_list.append(class_true_arr_list)
            val_class_score_arr_list.append(class_score_arr_list)

        val_loss_mean = np.mean(val_loss_mean)

        # iou_list: [L', 1]
        # class_true_arr_list, class_score_arr_list: [L', C]
        val_iou_list = np.vstack(val_iou_list)
        val_class_true_arr_list = np.vstack(val_class_true_arr_list)
        val_class_score_arr_list = np.vstack(val_class_score_arr_list)

        val_aps = get_aps(
            val_iou_list,
            val_class_true_arr_list,
            val_class_score_arr_list,
        )

        return val_loss_mean, val_aps

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
        train_aps_list = []

        val_loss_list = []
        val_aps_list = []

        max_map = 0

        for lr, num_epochs in zip(learning_rate_list, num_epochs_list):

            for epoch in range(1 + cum_epoch, num_epochs + 1 + cum_epoch):
                train_loss_mean, train_aps = self.train_one_step(
                    epoch, train_loader, lambda_coord, lambda_noobj, lr,
                )
                val_loss, val_aps = self.validate_one_step(
                    epoch, val_loader, lambda_coord, lambda_noobj,
                )

                print(
                    (
                        "Epoch: {} --> " +
                        "Training: (" +
                        "Loss Mean: {},  " +
                        "AP50: {},  " +
                        "AP75: {},  " +
                        "mAP: {}" +
                        ")    " +
                        "Validation: (" +
                        "Loss: {},  " +
                        "AP50: {},  " +
                        "AP75: {},  " +
                        "mAP: {}" +
                        ")"
                    )
                    .format(
                        epoch,
                        train_loss_mean,
                        train_aps[.50],
                        train_aps[.75],
                        train_aps["mAP"],
                        val_loss,
                        val_aps[.50],
                        val_aps[.75],
                        val_aps["mAP"],
                    )
                )

                train_loss_mean_list.append(train_loss_mean)
                train_aps_list.append(train_aps)

                val_loss_list.append(val_loss)
                val_aps_list.append(val_aps)

                if val_aps["mAP"] > max_map:
                    torch.save(
                        self.state_dict(),
                        os.path.join(
                            ckpt_path, "best_model.ckpt"
                        )
                    )
                    max_map = val_aps["mAP"]

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
                    "train_aps_list": train_aps_list,
                    "val_loss_list": val_loss_list,
                    "val_aps_list": val_aps_list,
                },
                f
            )

    def collate_fn_with_imgaug(self, batch):
        return self.collate_fn(batch, augmentation=True)

    def collate_fn(self, batch, augmentation=False, pad_val=-1):
        augmenter = iaa.Sequential([
            iaa.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            ),
            iaa.MultiplyHueAndSaturation(mul_saturation=(0.5, 1.5))
        ])
        resize = iaa.Resize({"height": self.h_in, "width": self.w_in})

        img_batch = []
        tgt_batch = []
        cls_idx_batch = []

        for img, bndbox_list in batch:
            img = np.array(img)

            if augmentation:
                img_aug, bndbox_aug_list = augmenter(
                    image=img, bounding_boxes=bndbox_list
                )

                bndbox_aug_list = (
                    bndbox_aug_list
                    .remove_out_of_image().clip_out_of_image()
                )

            else:
                img_aug, bndbox_aug_list = img, bndbox_list

            img_aug, bndbox_aug_list = resize(
                image=img_aug, bounding_boxes=bndbox_aug_list
            )

            tgt = [
                [
                    bndbox.x1,  # x1
                    bndbox.x2,  # x2
                    bndbox.y1,  # y1
                    bndbox.y2,  # y2
                    (bndbox.x1 + bndbox.x2) / 2,  # bx
                    (bndbox.y1 + bndbox.y2) / 2,  # by
                    bndbox.x2 - bndbox.x1,  # bw
                    bndbox.y2 - bndbox.y1,  # bh
                    (
                        ((bndbox.x1 + bndbox.x2) / 2) %
                        (self.w_in / self.S)
                    ) /
                    (self.w_in / self.S),  # bx_norm
                    (
                        ((bndbox.y1 + bndbox.y2) / 2) %
                        (self.h_in / self.S)
                    ) /
                    (self.h_in / self.S),  # by_norm
                    (bndbox.x2 - bndbox.x1) / self.w_in,  # bw_norm
                    (bndbox.y2 - bndbox.y1) / self.h_in,  # bh_norm
                ]
                for bndbox in bndbox_aug_list
            ]
            cls_idx = [
                self.cls2idx[bndbox.label] for bndbox in bndbox_aug_list
            ]

            if len(tgt) == 0:
                tgt.append(
                    [pad_val] * 12
                )
                cls_idx.append(0)

            img_batch.append(torch.tensor(img_aug).to(DEVICE))
            tgt_batch.append(torch.tensor(tgt).to(DEVICE))
            cls_idx_batch.append(torch.tensor(cls_idx).to(DEVICE))

        # img_batch: [N, W, H, 3]
        # tgt_batch: [N, L, 12]
        # mask_batch: [N, L]
        # cls_idx_batch: [N, L]
        img_batch = torch.stack(img_batch, dim=0)
        tgt_batch = pad_sequence(
            tgt_batch, batch_first=True, padding_value=pad_val
        )
        cls_idx_batch = pad_sequence(
            cls_idx_batch, batch_first=True, padding_value=0
        )
        mask_batch = (tgt_batch != pad_val).prod(-1)

        return img_batch, tgt_batch, cls_idx_batch, mask_batch
