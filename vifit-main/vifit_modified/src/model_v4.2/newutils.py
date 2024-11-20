#  ViFiT
# ===========

def compute_iou(pred_bbox, gt_bbox):
    """IoU for two bounding boxes."""
    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[0] + pred_bbox[2], gt_bbox[0] + gt_bbox[2])
    y2 = min(pred_bbox[1] + pred_bbox[3], gt_bbox[1] + gt_bbox[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = pred_bbox[2] * pred_bbox[3]
    gt_area = gt_bbox[2] * gt_bbox[3]
    union = pred_area + gt_area - intersection

    return intersection / union if union > 0 else 0


def adjust_bbox_for_camera(bbox, cam1_origin, n):
    """Adjusts bounding boxes for Camera 1's coordinate system."""
    x, y, w, h = bbox
    if x < cam1_origin[0] + n and y > cam1_origin[1] - n:
        # Adjust to Camera 1's origin
        return [x - cam1_origin[0], y - cam1_origin[1], w, h]
    else:
        # Mark as out of view for Camera 1
        return None


def prepare_virtual_camera_data(C, frames, n):
    """Sync data for virtual cameras."""
    cam1_frames, cam2_frames = [], []
    for frame in frames:
        cam1, cam2 = split_frame_into_cameras(frame, n)
        cam1_frames.append(cam1)
        cam2_frames.append(cam2)
    return cam1_frames, cam2_frames

def split_frame_into_cameras(frame, n):
    """Splits a frame into two virtual camera views."""
    height, width, _ = frame.shape
    n = min(n, height, width)

    # Camera 1: Bottom-left n x n region
    cam1 = frame[height-n:, :n]

    # Camera 2: Remaining part
    cam2 = frame.copy()
    cam2[height-n:, :n] = 0  # Mask Camera 1

    return cam1, cam2
