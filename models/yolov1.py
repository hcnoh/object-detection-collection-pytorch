import os
import pickle

import numpy as np
import torch
import torch.cuda
import torch.backends.mps
import imgaug.augmenters as iaa

from torch.nn import Module, Sequential, Flatten, Linear, ReLU, Dropout
from torch.optim import SGD

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
        y_pred_batch,
        y_tgt_batch,
        coord_batch,
        cls_tgt_batch,
        obj_mask_batch,
        lambda_coord,
        lambda_noobj,
    ):
        '''
            Args:
                N:
                    - the # of bounding boxes in the whole batch
                y_pred_batch:
                    - the model's prediction on the given targets
                    - [N, S, S, B * 5 + C]
                y_tgt_batch:
                    - the given targets
                    - [N, S, S, 4]
                coord_batch:
                    - the given coordinates for the all bounding boxes
                    - [N, S, S, 4]
                cls_tgt_batch:
                    - the given class targets as onehot vectors
                    - [N, C]
                obj_mask_batch:
                    - the mask to indicate the object is in each grid cell
                    - [N, S, S]
        '''
        w_in = self.w_in
        h_in = self.h_in

        S = self.S
        B = self.B
        C = self.C

        grid_cell_w = w_in / S
        grid_cell_h = h_in / S

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

        # bx_norm_tgt_batch, by_norm_tgt_batch,
        # bw_norm_tgt_batch, bh_norm_tgt_batch: [N, S, S, 1]
        bx_norm_tgt_batch = y_tgt_batch[:, :, :, 0].unsqueeze(-1)
        by_norm_tgt_batch = y_tgt_batch[:, :, :, 1].unsqueeze(-1)
        bw_norm_tgt_batch = y_tgt_batch[:, :, :, 2].unsqueeze(-1)
        bh_norm_tgt_batch = y_tgt_batch[:, :, :, 3].unsqueeze(-1)

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

        # obj_mask_batch: [N, S, S, 1]
        # noobj_mask_batch: [N, S, S, 1]
        obj_mask_batch = obj_mask_batch.unsqueeze(-1)
        noobj_mask_batch = (obj_mask_batch != 1)

        # i_arr: [1, S, 1, 1]
        # j_arr: [1, 1, S, 1]
        i_arr = (
            torch.arange(S).to(DEVICE)
            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        j_arr = (
            torch.arange(S).to(DEVICE)
            .unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )

        # bx_pred_batch, by_pred_batch,
        # bw_pred_batch, bh_pred_batch: [N, S, S, B]
        bx_pred_batch = (
            bx_norm_pred_batch * grid_cell_w +
            j_arr * grid_cell_w
        )
        by_pred_batch = (
            by_norm_pred_batch * grid_cell_h +
            i_arr * grid_cell_h
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
        # y1_tgt_batch, y2_tgt_batch: [N, S, S, 1]
        x1_tgt_batch = coord_batch[:, :, :, 0].unsqueeze(-1)
        x2_tgt_batch = coord_batch[:, :, :, 1].unsqueeze(-1)
        y1_tgt_batch = coord_batch[:, :, :, 2].unsqueeze(-1)
        y2_tgt_batch = coord_batch[:, :, :, 3].unsqueeze(-1)

        # iou_batch: [N, S, S, B]
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

        # max_iou_by_grid_cell_batch: [N, S, S, 1]
        max_iou_by_grid_cell_batch = (
            torch.max(iou_batch, dim=-1, keepdim=True).values
        )

        # responsible_mask_batch: [N, S, S, B]
        responsible_mask_batch = (iou_batch == max_iou_by_grid_cell_batch)
        responsible_mask_batch = (responsible_mask_batch * obj_mask_batch)

        # loss_xy: [N, S, S, B] -> [N]
        loss_xy = (
            (bx_norm_tgt_batch - bx_norm_pred_batch) ** 2 +
            (by_norm_tgt_batch - by_norm_pred_batch) ** 2
        )
        loss_xy = (loss_xy * responsible_mask_batch).sum(-1).sum(-1).sum(-1)

        # loss_wh: [N, S, S, B] -> [N]
        loss_wh = (
            (
                torch.sqrt(bw_norm_tgt_batch) - torch.sqrt(bw_norm_pred_batch)
            ) ** 2 +
            (
                torch.sqrt(bh_norm_tgt_batch) - torch.sqrt(bh_norm_pred_batch)
            ) ** 2
        )
        loss_wh = (loss_wh * responsible_mask_batch).sum(-1).sum(-1).sum(-1)

        # loss_conf: [N, S, S, B] -> [N]
        loss_conf = (iou_batch - conf_score_pred_batch) ** 2
        loss_conf = (loss_conf * obj_mask_batch).sum(-1).sum(-1).sum(-1)

        # loss_noobj: [N, S, S, B] -> [N]
        loss_noobj = (0 - conf_score_pred_batch) ** 2
        loss_noobj = (loss_noobj * noobj_mask_batch).sum(-1).sum(-1).sum(-1)

        # loss_cls: [N, S, S, C] -> [N]
        loss_cls = (
            cls_tgt_batch.unsqueeze(1).unsqueeze(1) -
            cond_cls_prob_pred_batch
        ) ** 2
        loss_cls = (loss_cls * obj_mask_batch).sum(-1).sum(-1).sum(-1)

        # loss: [N] -> []
        loss = (
            lambda_coord * loss_xy +
            lambda_coord * loss_wh +
            loss_conf +
            lambda_noobj * loss_noobj +
            loss_cls
        )
        loss = loss.mean()

        # cls_spec_conf_score_pred_batch: [N, S, S, B, C]
        cls_spec_conf_score_pred_batch = (
            cond_cls_prob_pred_batch.unsqueeze(-2) *
            conf_score_pred_batch.unsqueeze(-1)
        )

        # iou_batch: [N, S, S, B]
        # cls_tgt_batch: [N, C]
        # cls_score_batch: [N, S, S, B, C]
        iou_batch = iou_batch.detach().cpu().numpy()
        cls_tgt_batch = cls_tgt_batch.detach().cpu().numpy()
        cls_score_batch = cls_spec_conf_score_pred_batch.detach().cpu().numpy()

        return loss, iou_batch, cls_tgt_batch, cls_score_batch

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
        iou_batch = []
        cls_tgt_batch = []
        cls_score_batch = []
        img_id_batch = []

        dataset_size = len(data_loader.dataset)
        progress_size = 0

        for data in data_loader:
            # x_batch_one_step: [N, H, W, 3]
            # y_tgt_batch_one_step: [N, S, S, 4]
            # coord_batch_one_step: [N, S, S, 4]
            # cls_tgt_batch_one_step: [N, C]
            # obj_mask_batch_one_step: [N, S, S]
            # img_id_batch_one_step: [N]
            (
                x_batch_one_step,
                y_tgt_batch_one_step,
                coord_batch_one_step,
                cls_tgt_batch_one_step,
                obj_mask_batch_one_step,
                img_id_batch_one_step,
            ) = data

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

            (
                loss_one_step,
                iou_batch_one_step,
                cls_tgt_batch_one_step,
                cls_score_batch_one_step,
            ) = self.get_loss(
                N,
                y_pred_batch_one_step,
                y_tgt_batch_one_step,
                coord_batch_one_step,
                cls_tgt_batch_one_step,
                obj_mask_batch_one_step,
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
            iou_batch.append(iou_batch_one_step)
            cls_tgt_batch.append(cls_tgt_batch_one_step)
            cls_score_batch.append(cls_score_batch_one_step)
            img_id_batch.append(img_id_batch_one_step)

        loss_mean = np.mean(loss_mean)

        # iou_batch: [N, S, S, B]
        # cls_tgt_batch: [N, S, S, C]
        # cls_score_batch: [N, S, S, B, C]
        # img_id_batch: [N]
        iou_batch = np.vstack(iou_batch)
        cls_tgt_batch = np.vstack(cls_tgt_batch)
        cls_score_batch = np.vstack(cls_score_batch)
        img_id_batch = np.hstack(img_id_batch)

        aps = get_aps(
            iou_batch,
            cls_tgt_batch,
            cls_score_batch,
            img_id_batch,
        )

        return loss_mean, aps

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
                train_loss_mean, train_aps = self.execute_one_step(
                    epoch,
                    train_loader,
                    lambda_coord,
                    lambda_noobj,
                    lr,
                    train=True,
                )
                val_loss, val_aps = self.execute_one_step(
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
        img_id_batch = []

        for img_id, img, bndbox_list in batch:
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

            for bndbox in bndbox_aug_list:
                x = torch.tensor(img_aug).to(DEVICE)

                y_tgt = np.zeros(shape=[S, S, 4])
                coord = np.zeros(shape=[S, S, 4])
                cls_tgt = np.zeros(shape=[C])
                obj_mask = np.zeros(shape=[S, S])

                x1 = bndbox.x1
                x2 = bndbox.x2
                y1 = bndbox.y1
                y2 = bndbox.y2

                bx = (x1 + x2) / 2
                by = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1

                j = bx // grid_cell_w
                i = by // grid_cell_h

                bx_norm = (bx % grid_cell_w) / grid_cell_w
                by_norm = (by % grid_cell_h) / grid_cell_h
                bw_norm = bw / w_in
                bh_norm = bh / h_in

                y_tgt[i, j, 0] = bx_norm
                y_tgt[i, j, 1] = by_norm
                y_tgt[i, j, 2] = bw_norm
                y_tgt[i, j, 3] = bh_norm

                coord[i, j, 0] = x1
                coord[i, j, 1] = x2
                coord[i, j, 2] = y1
                coord[i, j, 3] = y2

                cls = bndbox.label
                cls_idx = self.cls2idx[cls]

                cls_tgt[cls_idx] = 1

                obj_mask[i, j] = 1

                y_tgt = torch.tensor(y_tgt).to(DEVICE)
                coord = torch.tensor(coord).to(DEVICE)
                cls_tgt = torch.tensor(cls_tgt).to(DEVICE)
                obj_mask = torch.tensor(obj_mask).to(DEVICE)

                x_batch.append(x)
                y_tgt_batch.append(y_tgt)
                coord_batch.append(coord)
                cls_tgt_batch.append(cls_tgt)
                obj_mask_batch.append(obj_mask)
                img_id_batch.append(img_id)

        # x_batch: [N, H, W, 3]
        # y_tgt_batch: [N, S, S, 4]
        # coord_batch: [N, S, S, 4]
        # cls_tgt_batch: [N, C]
        # obj_mask_batch: [N, S, S]
        # img_id_batch: [N]
        x_batch = torch.stack(x_batch, dim=0)
        y_tgt_batch = torch.stack(y_tgt_batch, dim=0)
        coord_batch = torch.stack(coord_batch, dim=0)
        cls_tgt_batch = torch.stack(cls_tgt_batch, dim=0)
        obj_mask_batch = torch.stack(obj_mask_batch, dim=0)
        img_id_batch = np.array(img_id_batch)

        return (
            x_batch,
            y_tgt_batch,
            coord_batch,
            cls_tgt_batch,
            obj_mask_batch,
            img_id_batch,
        )
