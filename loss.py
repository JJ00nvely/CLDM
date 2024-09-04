from torchvision.ops.boxes import box_area
from torchvision.ops import generalized_box_iou_loss
import torch

def validate_bbox_values(bbox):
    """
    Helper function to check if any of the bbox values are out of the expected range [0, 1].
    Raises ValueError if any value is out of range.
    """    
    if (bbox < 0).any():
        raise ValueError(f"BBox values are out of range: {bbox}")
    elif (bbox > 1).any():
        raise ValueError(f"BBox values are out of range: {bbox}")

def validate_bbox_values(bbox):
    """
    Helper function to check if any of the bbox values are out of expected range [0, 1].
    Raises ValueError if any value is out of range.
    """
    if (bbox < 0).any() or (bbox > 1).any():
        raise ValueError(f"BBox values are out of range: {bbox}")

def giou(bbox1, bbox2):
    if bbox1.size(0) != bbox2.size(0):
        raise ValueError(f"Batch size mismatch: bbox1 {bbox1.size(0)}, bbox2 {bbox2.size(0)}")
    

    def convert_to_xyxy(bbox):
        cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

        cx = (cx + 1) / 2
        cy = (cy + 1) / 2
        w  = (w + 1) / 2
        h  = (h + 1) / 2

        x1 = torch.clamp(cx - w / 2, 0, 1)
        y1 = torch.clamp(cy - h / 2, 0, 1)
        x2 = torch.clamp(cx + w / 2, 0, 1)
        y2 = torch.clamp(cy + h / 2, 0, 1)
        
        bbox_xyxy_noise = torch.stack([x1, y1, x2, y2], dim=1)
        validate_bbox_values(bbox_xyxy_noise)

        return bbox_xyxy_noise

    
    bbox1_xyxy = convert_to_xyxy(bbox1)
    bbox2_xyxy = convert_to_xyxy(bbox2)

    loss = generalized_box_iou_loss(bbox1_xyxy, bbox2_xyxy, reduction ='sum')
    loss = loss / bbox1.size(0)
    print(f"giou loss: {loss.item()}")
    return loss


    # def inter(bbox1, bbox2):
    #     x1 = torch.max(bbox1[:, 0], bbox2[:, 0])
    #     y1 = torch.max(bbox1[:, 1], bbox2[:, 1])
    #     x2 = torch.min(bbox1[:, 2], bbox2[:, 2])
    #     y2 = torch.min(bbox1[:, 3], bbox2[:, 3])
        
    #     inter_w = torch.clamp(x2 - x1, min=0)
    #     inter_h = torch.clamp(y2 - y1, min=0)
        
    #     return inter_w * inter_h
    
    # def union(bbox1, bbox2, inter_area):
    #     return box_area(bbox1) + box_area(bbox2) - inter_area
    
    # def enclosing_area(bbox1, bbox2):
    #     x1 = torch.min(bbox1[:, 0], bbox2[:, 0])
    #     y1 = torch.min(bbox1[:, 1], bbox2[:, 1])
    #     x2 = torch.max(bbox1[:, 2], bbox2[:, 2])
    #     y2 = torch.max(bbox1[:, 3], bbox2[:, 3])
        
    #     return (x2 - x1) * (y2 - y1)
    
    # inter_area = inter(bbox1_xyxy, bbox2_xyxy)
    # union_area = union(bbox1_xyxy, bbox2_xyxy, inter_area)
    # enc_area = enclosing_area(bbox1_xyxy, bbox2_xyxy)

    # iou = inter_area / (union_area + 1e-7)
    # giou = iou - (enc_area - union_area) / (enc_area + 1e-7)

    # print(f"inter_area: {inter_area.mean().item()}")
    # print(f"union_area: {union_area.mean().item()}")
    # print(f"enc_area: {enc_area.mean().item()}")
    # print(f"iou: {iou.mean().item()}")
    # print(f"giou: {giou.mean().item()}")

    # return ((1 -giou).sum())/bbox1.size(0)

import torch

def bbox_pair_tv_loss(bboxes, frames= 16, tv_weight=1.0):
    """
    Compute the TV loss for a sequence of bounding boxes.

    Args:
        bboxes: Tensor of shape (num_bboxes, 4)
                where each bbox is represented as (x, y, w, h).
        tv_weight: Weight for the TV loss.

    Returns:
        loss: Scalar representing the total variation loss for the bounding boxes.
    """
    nums = bboxes.shape[0]//frames
    total_loss = 0.0
    
    for i in range(nums):
        start = i * frames
        end = (i+1)* frames
        sequence_box = bboxes[start:end,:]
    # Calculate differences between each pair of consecutive bounding boxes
        diff = torch.abs(sequence_box[1:, :] - sequence_box[:-1, :])
        sequence_loss = diff.sum(dim=1).mean()
    # Sum differences across the bbox dimensions (x, y, w, h) and mean
        total_loss +=sequence_loss
        
    total_loss /= nums

    total_loss = tv_weight * total_loss
    
    return total_loss


    