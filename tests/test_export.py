"""Class for testing prediction and export scripts"""
import numpy as np

from script.convert_xml import polygon_to_string
from script.transkribus_export import prediction_to_polygons
from src.news_seg import predict


class TestClassExport:
    """Class for testing prediction and export scripts"""

    def test_process_prediction(self):
        """Function for testing prediction argmax and threshold handling.
        Each tripple of data represents probabilities for 3 possible classes.
        If the maximum is above the threshold, the result should contain that class label.
        Otherwise, it is always class 0."""
        data = np.transpose(
            np.array(
                [
                    [[0.1, 0.5, 0.4], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]],
                    [[0.0, 0.6, 0.4], [0.05, 0.05, 0.9], [0.01, 0.59, 0.4]],
                ]
            ),
            (2, 0, 1),
        )
        ground_truth = np.array([[0, 1, 1], [1, 2, 0]])

        result = predict.process_prediction(data, 0.6)
        assert np.all(result == ground_truth)

    def test_polygon_to_string(self):
        """Tests polygon list conversion to a coordinate string that transkribus can handle."""
        data = [19.0, 20.0, 1.0, 4.0, 5.5, 10.5, 20.0, 30.0]
        ground_truth = "19,20 1,4 5,10 20,30"

        assert polygon_to_string(data) == ground_truth

    def test_prediction_to_polygons(self):
        """Tests prediction conversion to a polygon list. Background pixels will not be converted to a polygon"""
        data = np.array([[0, 0, 3, 3, 3], [0, 0, 3, 3, 1], [1, 1, 1, 1, 1]])
        ground_truth = {
            1: [[4.0, 2.5, -0.5, 2.0, 4.0, 0.5, 4.0, 2.5]],
            3: [[3.0, 1.5, 1.5, 1.0, 2.0, -0.5, 4.5, 0.0, 3.0, 1.5]],
        }
        assert prediction_to_polygons(data) == ground_truth