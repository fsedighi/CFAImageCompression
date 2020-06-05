import unittest

from Utils.DatasetUtils import DatasetUtils


class test_DatasetUtils(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.datasetUtils = DatasetUtils()

    def test_generateCFAImages(self):
        rgbImages = self.datasetUtils.loadFoodDataset()
        cfaImages, image_size = self.datasetUtils.convertDatasetToCFA(rgbImages)
        self.assertIsNotNone(cfaImages)
