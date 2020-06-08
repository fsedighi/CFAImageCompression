import unittest

from Utils.DataUtils import DataUtils


class test_DatasetUtils(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.datasetUtils = DataUtils()

    def test_generateCFAImages(self):
        rgbImages = self.datasetUtils.loadKodakDataset()
        cfaImages, image_size = self.datasetUtils.convertDatasetToCFA(rgbImages)
        self.assertIsNotNone(cfaImages)
