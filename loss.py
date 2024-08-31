from torchvision.ops.boxes import box_area
import torch


def giou(bbox1, bbox2):
    if bbox1.size(0) != bbox2.size(0):
        raise ValueError(f"Batch size mismatch: bbox1 {bbox1.size(0)}, bbox2 {bbox2.size(0)}")

    def convert_to_xyxy(bbox):
        
        cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        cx = (cx+1)/2
        cy = (cy+1)/2
        w  = (w+1)/2
        h  = (h+1)/2
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    bbox1_xyxy = convert_to_xyxy(bbox1)
    bbox2_xyxy = convert_to_xyxy(bbox2)

    def inter(bbox1, bbox2):
        x1 = torch.max(bbox1[:, 0], bbox2[:, 0])
        y1 = torch.max(bbox1[:, 1], bbox2[:, 1])
        x2 = torch.min(bbox1[:, 2], bbox2[:, 2])
        y2 = torch.min(bbox1[:, 3], bbox2[:, 3])
        
        inter_w = torch.clamp(x2 - x1, min=0)
        inter_h = torch.clamp(y2 - y1, min=0)
        
        return inter_w * inter_h
    
    def union(bbox1, bbox2, inter_area):
        return box_area(bbox1) + box_area(bbox2) - inter_area
    
    def enclosing_area(bbox1, bbox2):
        x1 = torch.min(bbox1[:, 0], bbox2[:, 0])
        y1 = torch.min(bbox1[:, 1], bbox2[:, 1])
        x2 = torch.max(bbox1[:, 2], bbox2[:, 2])
        y2 = torch.max(bbox1[:, 3], bbox2[:, 3])
        
        return (x2 - x1) * (y2 - y1)
    
    inter_area = inter(bbox1_xyxy, bbox2_xyxy)
    union_area = union(bbox1_xyxy, bbox2_xyxy, inter_area)
    enc_area = enclosing_area(bbox1_xyxy, bbox2_xyxy)

    iou = inter_area / (union_area + 1e-7)
    giou = iou - (enc_area - union_area) / (enc_area + 1e-7)
    
    return giou