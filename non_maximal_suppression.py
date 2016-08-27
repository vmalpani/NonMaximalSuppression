def intersection_over_union((left1, top1, width1, height1, score1), (left2, top2, width2, height2, score2)):
    """
    Calculates the intersection over union measure of two bounding boxes 
    """
    right1 = left1 + width1
    bottom1 = top1 + height1
 
    right2 = left2 + width2
    bottom2 = top2 + height2

    intersect_width = max(0, min(right1, right2) - max(left1, left2))
    intersect_height = max(0, min(bottom1, bottom2) - max(top1, top2))
    intersection = intersect_width * intersect_height

    area1 = width1 * height1
    area2 = width2 * height2
    union = area1 + area2 - intersection

    return intersection/float(union)


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
    detections.sort(key=lambda x:x[4], reverse=True)
    final_detections = [detections.pop(0)]
    while len(detections) > 0:
	bbox = detections.pop(0)
        for final_bbox in final_detections:
            iou = intersection_over_union(bbox, final_bbox)
	    assert iou >= 0 and iou <= 1
	    if iou > threshold:
               break
        else:
            final_detections.append(bbox)
    return final_detections
    


if __name__ == "__main__":
    detections = [(11, 11, 24, 24, 0.75), (10, 11, 20, 20, 0.8), (11, 9, 24, 24, 0.7), (40, 42, 20, 20, 0.6)]
    print 'Detections: ' + ', '.join(map(str, detections))
    final_detections = non_maximal_suppression(detections, threshold=0.5)
    print 'Pruned Detections: ' + ', '.join(map(str, final_detections))
