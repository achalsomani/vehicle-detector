from torchvision.ops import nms

def apply_nms_to_predictions(boxes, labels, scores, iou_threshold=0.5, score_threshold=0.1):
    mask = scores > score_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    if len(boxes) == 0:
        return boxes.new_zeros((0, 4)), labels.new_zeros(0), scores.new_zeros(0)
    
    _, sorted_indices = scores.sort(descending=True)
    boxes = boxes[sorted_indices]
    labels = labels[sorted_indices]
    scores = scores[sorted_indices]
    
    keep = nms(boxes, scores, iou_threshold=iou_threshold)
    
    return boxes[keep], labels[keep], scores[keep]
