import numpy as np


def intersection_over_union(src_bbox, target_bboxes):
    """
    Vectorized calculation of intersection over union measure of a source bounding box 
    with all the target bounding boxes
    """
    src_box_left = src_bbox[0]
    src_box_top = src_bbox[1]
    src_box_right = src_bbox[0] + src_bbox[2]
    src_box_bottom = src_bbox[1] + src_bbox[3]

    target_bboxes_left = target_bboxes[:,0]
    target_bboxes_top = target_bboxes[:,1]
    target_bboxes_right = target_bboxes[:,0] + target_bboxes[:,2]
    target_bboxes_bottom = target_bboxes[:,1] + target_bboxes[:,3]

    intersect_width = np.maximum(0, np.minimum(target_bboxes_right, src_box_right) - np.maximum(target_bboxes_left, src_box_left))
    intersect_height = np.maximum(0, np.minimum(target_bboxes_bottom, src_box_bottom) - np.maximum(target_bboxes_top, src_box_top))
    intersection = intersect_width * intersect_height

    area_src = src_bbox[2] * src_bbox[3]
    area_target = target_bboxes[:,2] * target_bboxes[:,3]
    union = area_src + area_target - intersection

    return np.divide(intersection, union)


def non_maximal_suppression(detections, threshold=0.5):
    """
    Performs non-maximal suppression using intersection over union measure and picking the max score detections
    Bounding box format: x, y, w, h, score
    Input: [(11, 11, 24, 24, 0.75), (10, 11, 20, 20, 0.8), (11, 9, 24, 24, 0.7), (40, 42, 20, 20, 0.6)]
    Output: [(10, 11, 20, 20, 0.8), (40, 42, 20, 20, 0.6)]
    """
    if len(detections) < 2:
        raise ValueError("Need at least two bounding boxes")

    # fourth index corresponds to detection score
    detections = detections[detections[:,4].argsort()[::-1]]

    final_detections = np.array([detections[0]])
    for bbox in detections:
	iou = intersection_over_union(bbox, final_detections)
        assert (iou >= 0).all() and (iou <= 1).all()
        overlap_idxs = np.where(iou > 0.5)
        if overlap_idxs[0].size == 0:
            final_detections = np.vstack((final_detections, bbox))
    return final_detections
    


if __name__ == "__main__":
    detections = np.array([(11, 11, 24, 24, 0.75), (10, 11, 20, 20, 0.8), (11, 9, 24, 24, 0.7), (40, 42, 20, 20, 0.6)])
    print 'Detections:\n' + np.array_str(detections)
    final_detections = non_maximal_suppression(detections, threshold=0.5)
    print '\nPruned Detections:\n' + np.array_str(final_detections)
