def intersection_over_union(bbox1, bbox2):
    """
    Calculates the intersection over union measure of two bounding boxes 
    """
    bbox1_x1 = bbox1[0] 
    bbox1_x2 = bbox1[0] + bbox1[2]
    bbox1_y1 = bbox1[1]
    bbox1_y2 = bbox1[1] + bbox1[3]
 
    bbox2_x1 = bbox2[0] 
    bbox2_x2 = bbox2[0] + bbox2[2]
    bbox2_y1 = bbox2[1]
    bbox2_y2 = bbox2[1] + bbox2[3]

    intersect_width = max(0, min(bbox1_x2, bbox2_x2) - max(bbox1_x1, bbox2_x1))
    intersect_height = max(0, min(bbox1_y2, bbox2_y2) - max(bbox1_y1, bbox2_y1))
    intersection = intersect_width * intersect_height

    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union = area1 + area2 - intersection

    return intersection/float(union)


def non_maximal_suppression(detections, threshold=0.5):
    """
    Performs non-maximal suppression using intersection over union measure and picking the max score detections
    Bounding box format: x, y, w, h
    Input: [(11, 11, 24, 24, 0.75), (10, 11, 20, 20, 0.8), (11, 9, 24, 24, 0.7), (40, 42, 20, 20, 0.6)]
    Output: [(10, 11, 20, 20, 0.8), (40, 42, 20, 20, 0.6)]
    """
    if len(detections) < 2:
        raise ValueError("Need at least two bounding boxes")

    # fourth index corresponds to detection score
    sorted_detections = sorted(detections,key=lambda x:x[4], reverse=True)
    final_detections = [sorted_detections[0]]
    sorted_detections_copy = list(sorted_detections[1:])
    while len(sorted_detections_copy) > 0:
	for idx, bbox in enumerate(sorted_detections[1:]):
            flag_overlap = False
	    for final_bbox in final_detections:
		iou = intersection_over_union(bbox, final_bbox)
		assert iou >= 0 and iou <= 1
		if iou > threshold:
                    flag_overlap = True
                    break
            if not flag_overlap:
                final_detections.append(bbox)
            sorted_detections_copy.remove(bbox)
    return final_detections
    


if __name__ == "__main__":
    detections = [(11, 11, 24, 24, 0.75), (10, 11, 20, 20, 0.8), (11, 9, 24, 24, 0.7), (40, 42, 20, 20, 0.6)]
    print 'Detections: ' + ', '.join(map(str, detections))
    final_detections = non_maximal_suppression(detections, threshold=0.5)
    print 'Pruned Detections: ' + ', '.join(map(str, final_detections))
