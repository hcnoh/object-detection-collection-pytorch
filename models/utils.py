import numpy as np
import torch


def get_iou(
    x1_batch,
    y1_batch,
    x2_batch,
    y2_batch,
    x1_hat_batch,
    y1_hat_batch,
    x2_hat_batch,
    y2_hat_batch,
):
    '''
        Returns the IOU batch from x1, y1, x2, y2, x1_hat, y1_hat, x2_hat,
        y2_hat batches

        Args:
            x1_batch:
                - [N, ...]
            y1_batch:
                - [N, ...]
            x2_batch:
                - [N, ...]
            y2_batch:
                - [N, ...]
            x1_hat_batch:
                - [N, ...]
            y1_hat_batch:
                - [N, ...]
            x2_hat_batch:
                - [N, ...]
            y2_hat_batch:
                - [N, ...]

        Returns:
            iou_batch:
                - [N, ...]
    '''
    intsec_x1_batch = torch.maximum(x1_batch, x1_hat_batch)
    intsec_y1_batch = torch.maximum(y1_batch, y1_hat_batch)
    intsec_x2_batch = torch.minimum(x2_batch, x2_hat_batch)
    intsec_y2_batch = torch.minimum(y2_batch, y2_hat_batch)

    intsec_batch = (
        torch.clamp(intsec_x2_batch - intsec_x1_batch, min=0) *
        torch.clamp(intsec_y2_batch - intsec_y1_batch, min=0)
    )

    union_batch = (
        (x2_batch - x1_batch) * (y2_batch - y1_batch) +
        (x2_hat_batch - x1_hat_batch) * (y2_hat_batch - y1_hat_batch) -
        intsec_batch
    )

    iou_batch = intsec_batch / (union_batch + 1e-6)

    return iou_batch


def nms(
    x1_batch,
    y1_batch,
    x2_batch,
    y2_batch,
    conf_score_batch,
    cls_spec_conf_score_batch,
    conf_score_thre=0.6,
    iou_thre=0.5,
):
    '''
        Args:
            x1_batch:
                - torch.Tensor
                - [N, ...]
            y1_batch:
                - torch.Tensor
                - [N, ...]
            x2_batch:
                - torch.Tensor
                - [N, ...]
            y2_batch:
                - torch.Tensor
                - [N, ...]
            conf_score_batch:
                - torch.Tensor
                - [N, ...]
            cls_spec_conf_score_batch:
                - torch.Tensor
                - [N, ..., num_cls]
    '''
    C = cls_spec_conf_score_batch.shape[-1]

    '''conf_score_mask_batch: [N, ...]'''
    conf_score_mask_batch = (conf_score_batch >= conf_score_thre).bool()

    '''x1_batch, x2_batch, y1_batch, y2_batch, conf_score_batch: [M]'''
    x1_batch = torch.masked_select(x1_batch, conf_score_mask_batch)
    x2_batch = torch.masked_select(x2_batch, conf_score_mask_batch)
    y1_batch = torch.masked_select(y1_batch, conf_score_mask_batch)
    y2_batch = torch.masked_select(y2_batch, conf_score_mask_batch)
    conf_score_batch = torch.masked_select(
        conf_score_batch, conf_score_mask_batch
    )
    cls_spec_conf_score_batch = torch.masked_select(
        cls_spec_conf_score_batch,
        conf_score_mask_batch.unsqueeze(-1),
    ).reshape([-1, C])

    '''for sorting the coordinates.'''
    conf_score_batch, sorted_indices = torch.sort(
        conf_score_batch, descending=True
    )

    x1_batch = x1_batch[sorted_indices]
    x2_batch = x2_batch[sorted_indices]
    y1_batch = y1_batch[sorted_indices]
    y2_batch = y2_batch[sorted_indices]

    cls_spec_conf_score_batch = cls_spec_conf_score_batch[sorted_indices]

    i = 0
    while i < len(conf_score_batch) - 1:
        '''x1_tgt, x2_tgt, y1_tgt, y2_tgt: [1]'''
        x1_tgt = x1_batch[i:i + 1]
        x2_tgt = x2_batch[i:i + 1]
        y1_tgt = y1_batch[i:i + 1]
        y2_tgt = y2_batch[i:i + 1]

        '''iou_batch: [M - i]'''
        iou_batch = get_iou(
            x1_tgt,
            x2_tgt,
            y1_tgt,
            y2_tgt,
            x1_batch[i + 1:],
            x2_batch[i + 1:],
            y1_batch[i + 1:],
            y2_batch[i + 1:],
        )

        '''iou_mask: [M - i]'''
        iou_mask = (iou_batch < iou_thre).bool()

        '''iou_mask: [M]'''
        iou_mask = torch.cat(
            [torch.tensor([True] * (i + 1)), iou_mask], dim=-1
        )
        iou_mask = iou_mask.bool()

        '''x1_batch, x2_batch, y1_batch, y2_batch, conf_score_batch: [M']'''
        x1_batch = torch.masked_select(x1_batch, iou_mask)
        x2_batch = torch.masked_select(x2_batch, iou_mask)
        y1_batch = torch.masked_select(y1_batch, iou_mask)
        y2_batch = torch.masked_select(y2_batch, iou_mask)
        conf_score_batch = torch.masked_select(conf_score_batch, iou_mask)
        cls_spec_conf_score_batch = torch.masked_select(
            cls_spec_conf_score_batch, iou_mask.unsqueeze(-1),
        ).reshape([-1, C])

        i += 1

    return (
        x1_batch,
        y1_batch,
        x2_batch,
        y2_batch,
        conf_score_batch,
        cls_spec_conf_score_batch,
    )


def cummax(x):
    return np.array([np.max(x[:i + 1]) for i in range(len(x))])


def get_aps(
    iou_batch,
    cls_tgt_batch,
    cls_score_batch,
    bbox_img_id_batch,
    level_list=[.50, .55, .60, .65, .70, .75, .80, .85, .90, .95],
):
    '''
        Args:
            iou_batch: [N, S, S, B]
            cls_tgt_batch: [N, C]
            cls_score_batch: [N, S, S, B, C]
            img_id_batch: [N]
    '''
    # N = iou_batch.shape[0]
    # S = iou_batch.shape[1]
    # B = iou_batch.shape[-1]
    C = cls_tgt_batch.shape[-1]

    aps = {}

    for level in level_list:

        aps_by_class = []

        for cls_idx in range(C):
            '''selected_cls_indices: [M]'''
            selected_cls_indices = np.where(cls_tgt_batch[:, cls_idx] == 1)[0]

            '''
            selected_iou_batch: [M, S, S, B]
            selected_cls_tgt_batch: [M]
            selected_cls_score_batch: [M, S, S, B]
            selected_img_id_batch: [M]
            '''
            selected_iou_batch = iou_batch[selected_cls_indices]
            # selected_cls_tgt_batch = cls_tgt_batch[
            #     selected_cls_indices, cls_idx
            # ]
            selected_cls_score_batch = cls_score_batch[
                selected_cls_indices, :, :, :, cls_idx
            ]
            selected_img_id_batch = bbox_img_id_batch[selected_cls_indices]

            num_gt = selected_iou_batch.shape[0]

            fp_list = []
            tp_list = []
            score_list = []

            for img_id in np.unique(selected_img_id_batch):
                '''one_img_indices: [L]'''
                one_img_indices = np.where(selected_img_id_batch == img_id)

                '''
                one_img_iou_batch: [L, S, S, B]
                one_img_cls_score_batch: [L, S, S, B] -> [S, S, B]
                '''
                one_img_iou_batch = selected_iou_batch[
                    one_img_indices
                ]
                one_img_cls_score_batch = selected_cls_score_batch[
                    one_img_indices
                ]
                one_img_cls_score_batch = one_img_cls_score_batch[0]

                '''one_img_max_iou_by_grid_cell_batch: [S, S, B]'''
                one_img_max_iou_by_grid_cell_batch = np.max(
                    one_img_iou_batch, axis=0
                )

                '''fp, tp: [S, S, B]'''
                fp = (one_img_max_iou_by_grid_cell_batch < level)
                tp = (one_img_max_iou_by_grid_cell_batch >= level)

                fp_list.append(fp.flatten())
                tp_list.append(tp.flatten())
                score_list.append(one_img_cls_score_batch.flatten())

            '''fp_list, tp_list, score_list: [M]'''
            fp_list = np.hstack(fp_list)
            tp_list = np.hstack(tp_list)
            score_list = np.hstack(score_list)

            sorted_indices = np.argsort(score_list)[::-1]

            sorted_fp_list = fp_list[sorted_indices]
            sorted_tp_list = tp_list[sorted_indices]
            # sorted_score_list = score_list[sorted_indices]

            cum_fp_list = np.cumsum(sorted_fp_list)
            cum_tp_list = np.cumsum(sorted_tp_list)

            prec_list = cum_tp_list / (cum_tp_list + cum_fp_list)
            rec_list = cum_tp_list / num_gt

            reverse_cummax_prec_list = cummax(prec_list[::-1])[::-1]

            rec_diff_list = rec_list - np.hstack([[0], rec_list[:-1]])

            # print("==============================================")
            # print(cls_idx)
            # print("sorted_score_list", sorted_score_list)
            # print("prec_list", prec_list)
            # print("rec_list", rec_list)
            # print("reverse_cummax_prec_list", reverse_cummax_prec_list)
            # print("rec_diff_list", rec_diff_list)

            ap = np.sum(reverse_cummax_prec_list * rec_diff_list)

            # print("ap", ap)

            aps_by_class.append(ap)

        aps["APs by Class"] = aps_by_class
        aps[level] = np.mean(aps_by_class)

    mean_ap = np.mean([aps[level] for level in level_list])
    aps["mAP"] = mean_ap

    return aps
