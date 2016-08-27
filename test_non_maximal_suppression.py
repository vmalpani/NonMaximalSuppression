import unittest
import non_maximal_suppression as nms


class NMSTests(unittest.TestCase):

    def test_non_maximal_suppression_multiple_overlap(self):
        detections = [(11, 11, 24, 24, 0.75), (10, 11, 20, 20, 0.8), (11, 9, 24, 24, 0.7), (40, 42, 20, 20, 0.6)]
        pruned_detections = nms.non_maximal_suppression(detections)
        self.assertEqual(pruned_detections, [(10, 11, 20, 20, 0.8), (40, 42, 20, 20, 0.6)])

    def test_non_maximal_suppression_no_overlap(self):
        detections = [(11, 11, 24, 24, 0.75), (40, 42, 20, 20, 0.6)]
        pruned_detections = nms.non_maximal_suppression(detections)
        self.assertEqual(pruned_detections, [(11, 11, 24, 24, 0.75), (40, 42, 20, 20, 0.6)])

    def test_non_maximal_suppression_just_under_threshold_overlap(self):
        detections = [(5, 10, 21, 21, 0.75), (12, 10, 21, 21, 0.6)]
        pruned_detections = nms.non_maximal_suppression(detections)
        self.assertEqual(pruned_detections, [(5, 10, 21, 21, 0.75), (12, 10, 21, 21, 0.6)])

    def test_non_maximal_suppression_just_over_threshold_overlap(self):
        detections = [(5, 10, 21, 21, 0.75), (11, 10, 21, 21, 0.6)]
        pruned_detections = nms.non_maximal_suppression(detections)
        self.assertEqual(pruned_detections, [(5, 10, 21, 21, 0.75)])

    def test_non_maximal_suppression_exception(self):
        detections = [(11, 11, 24, 24, 0.75)]
        with self.assertRaises(ValueError) as context:
            nms.non_maximal_suppression(detections)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
