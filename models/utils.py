import numpy as np
import torch


def get_iou(
    coord1_batch,
    coord2_batch,
    numpy=False,
):
    '''
        Returns the IOU batch from coord1_batch and coord2_batch

        coord: [x1, y1, x2, y2]

        Args:
            coord1_batch:
                - [..., 4]
            coord2_batch:
                - [..., 4]

        Returns:
            iou_batch:
                - [...]
    '''
    x1_batch = coord1_batch[..., 0]
    y1_batch = coord1_batch[..., 1]
    x2_batch = coord1_batch[..., 2]
    y2_batch = coord1_batch[..., 3]

    x1_hat_batch = coord2_batch[..., 0]
    y1_hat_batch = coord2_batch[..., 1]
    x2_hat_batch = coord2_batch[..., 2]
    y2_hat_batch = coord2_batch[..., 3]

    if numpy:
        intsec_x1_batch = np.maximum(x1_batch, x1_hat_batch)
        intsec_y1_batch = np.maximum(y1_batch, y1_hat_batch)
        intsec_x2_batch = np.minimum(x2_batch, x2_hat_batch)
        intsec_y2_batch = np.minimum(y2_batch, y2_hat_batch)

        intsec_batch = (
            np.clip(intsec_x2_batch - intsec_x1_batch, a_min=0, a_max=None) *
            np.clip(intsec_y2_batch - intsec_y1_batch, a_min=0, a_max=None)
        )

    else:
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
    bbox_coord_batch,
    conf_score_batch,
    cls_spec_conf_score_batch,
    conf_score_thre=0.9,
    iou_thre=0.5,
):
    '''
        coord: [x1, y1, x2, y2]

        Args:
            bbox_coord_batch:
                - torch.Tensor
                - [..., 4]
            conf_score_batch:
                - torch.Tensor
                - [...]
            cls_spec_conf_score_batch:
                - torch.Tensor
                - [..., num_cls]
    '''
    num_cls = cls_spec_conf_score_batch.shape[-1]

    '''conf_score_mask_batch: [...]'''
    conf_score_mask_batch = (conf_score_batch >= conf_score_thre).bool()

    '''
    bbox_coord_batch: [M, 4]
    conf_score_batch: [M]
    cls_spec_conf_score_batch: [M, num_cls]
    '''
    bbox_coord_batch = (
        torch.masked_select(
            bbox_coord_batch, conf_score_mask_batch.unsqueeze(-1)
        ).reshape([-1, 4])
    )
    conf_score_batch = torch.masked_select(
        conf_score_batch, conf_score_mask_batch
    )
    cls_spec_conf_score_batch = (
        torch.masked_select(
            cls_spec_conf_score_batch,
            conf_score_mask_batch.unsqueeze(-1),
        )
        .reshape([-1, num_cls])
    )

    '''for sorting the coordinates.'''
    conf_score_batch, sorted_indices = torch.sort(
        conf_score_batch, descending=True
    )

    bbox_coord_batch = bbox_coord_batch[sorted_indices]

    cls_spec_conf_score_batch = cls_spec_conf_score_batch[sorted_indices]

    i = 0
    while i < len(conf_score_batch) - 1:
        '''iou_batch: [M - i]'''
        iou_batch = get_iou(
            bbox_coord_batch[i:i + 1],
            bbox_coord_batch[i + 1:],
        )

        '''iou_mask: [M - i]'''
        iou_mask = (iou_batch < iou_thre).bool()

        '''iou_mask: [M]'''
        iou_mask = torch.cat(
            [torch.tensor([True] * (i + 1)), iou_mask], dim=-1
        )
        iou_mask = iou_mask.bool()

        '''
        coord_batch: [M', 4]
        conf_score_batch: [M']
        cls_spec_conf_score_batch: [M', num_cls]
        '''
        bbox_coord_batch = (
            torch.masked_select(bbox_coord_batch, iou_mask.unsqueeze(-1))
            .reshape([-1, 4])
        )
        conf_score_batch = torch.masked_select(conf_score_batch, iou_mask)
        cls_spec_conf_score_batch = (
            torch.masked_select(
                cls_spec_conf_score_batch, iou_mask.unsqueeze(-1),
            )
            .reshape([-1, num_cls])
        )

        i += 1

    return (
        bbox_coord_batch,
        conf_score_batch,
        cls_spec_conf_score_batch,
    )


def cummax(x, axis=-1):
    return np.array([np.max(x[:i + 1], axis=axis) for i in range(len(x))])


def evaluate_model(
    model,
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
        c: [] for c in model.cls_list
    }
    cls_spec_fp_list = {
        c: [] for c in model.cls_list
    }
    cls_spec_conf_score_list = {
        c: [] for c in model.cls_list
    }
    cls_spec_num_gt_list = {
        c: 0 for c in model.cls_list
    }

    dataset_size = len(dataset)
    progress_size = 0

    for _, img, annot in dataset:
        progress_size += 1
        # if progress_size > 1000:
        #     break

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

        for cls in model.cls_list:
            cls_spec_num_gt_list[cls] += np.sum(cls_batch == cls)

        annot_pred = model.detect(img, conf_score_thre, iou_thre)

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

    evaluation_result = {}
    evaluation_result["level_list"] = level_list

    for cls in model.cls_list:
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

        '''sorted_prec_list_reverse_cummax: [num_bbox, num_level]'''
        '''sorted_rec_diff_list: [num_bbox, num_level]'''
        sorted_prec_list_reverse_cummax = (
            cummax(sorted_prec_list[::-1], axis=0)[::-1]
        )

        sorted_rec_diff_list = np.zeros_like(sorted_rec_list)
        sorted_rec_diff_list[1:] = sorted_rec_list[:-1]
        sorted_rec_diff_list = sorted_rec_list - sorted_rec_diff_list

        '''ap: []'''
        ap = np.sum(
            sorted_prec_list_reverse_cummax * sorted_rec_diff_list, axis=0
        )

        evaluation_result[cls] = ap

    # print(evaluation_result["level_list"])
    # for cls in model.cls_list:
    #     print("Class: {}".format(cls))
    #     print("APs: {}".format(evaluation_result[cls]))

    return evaluation_result


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
