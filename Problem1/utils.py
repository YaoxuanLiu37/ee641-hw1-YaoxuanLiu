import torch
import numpy as np

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    
    anchors_all = []
    for (H, W), scales in zip(feature_map_sizes, anchor_scales):
        stride_y = image_size / float(H)
        stride_x = image_size / float(W)
        ys = (torch.arange(H, dtype=torch.float32) + 0.5) * stride_y
        xs = (torch.arange(W, dtype=torch.float32) + 0.5) * stride_x
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
        centers = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # [H*W,2]
        per_scale = []
        for s in scales:  # 1:1 aspect only
            w = torch.full((centers.size(0),), float(s), dtype=torch.float32)
            h = torch.full((centers.size(0),), float(s), dtype=torch.float32)
            x1 = (centers[:, 0] - 0.5 * w).clamp(0, image_size - 1)
            y1 = (centers[:, 1] - 0.5 * h).clamp(0, image_size - 1)
            x2 = (centers[:, 0] + 0.5 * w).clamp(0, image_size - 1)
            y2 = (centers[:, 1] + 0.5 * h).clamp(0, image_size - 1)
            per_scale.append(torch.stack([x1, y1, x2, y2], dim=1))
        anchors = torch.cat(per_scale, dim=0)  # [H*W*num_anchors,4]
        anchors_all.append(anchors)
    return anchors_all

def compute_iou(boxes1, boxes2):
    
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), dtype=torch.float32)

    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = torch.max(x11[:, None], x21[None, :])
    inter_y1 = torch.max(y11[:, None], y21[None, :])
    inter_x2 = torch.min(x12[:, None], x22[None, :])
    inter_y2 = torch.min(y12[:, None], y22[None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return iou

def match_anchors_to_targets(anchors, target_boxes, target_labels, 
                            pos_threshold=0.5, neg_threshold=0.3):
    device = anchors.device
    A = anchors.size(0)
    matched_labels = torch.zeros((A,), dtype=torch.long, device=device)
    matched_boxes = torch.zeros((A, 4), dtype=torch.float32, device=device)
    if target_boxes.numel() == 0:
        pos_mask = torch.zeros((A,), dtype=torch.bool, device=device)
        neg_mask = torch.ones((A,), dtype=torch.bool, device=device)
        return matched_labels, matched_boxes, pos_mask, neg_mask
    iou_mat = compute_iou(anchors, target_boxes.to(device))
    max_iou, max_idx = iou_mat.max(dim=1)
    pos_mask = (max_iou >= pos_threshold)
    neg_mask = (max_iou < neg_threshold)
    ignore_mask = (~pos_mask) & (~neg_mask)
    tb = target_boxes.to(device)
    tl = target_labels.to(device).long()
    matched_boxes[pos_mask] = tb[max_idx[pos_mask]]
    matched_labels[pos_mask] = tl[max_idx[pos_mask]] + 1
    best_anchor_for_gt = iou_mat.argmax(dim=0)
    matched_boxes[best_anchor_for_gt] = tb
    matched_labels[best_anchor_for_gt] = tl + 1
    pos_mask[best_anchor_for_gt] = True
    neg_mask[best_anchor_for_gt] = False
    ignore_mask[best_anchor_for_gt] = False
    return matched_labels, matched_boxes, pos_mask, neg_mask
