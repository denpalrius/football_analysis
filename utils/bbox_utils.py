def get_bbox_center(bbox) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox) -> int:
    x1, _, x2, _ = bbox
    return int(x2) - int(x1)