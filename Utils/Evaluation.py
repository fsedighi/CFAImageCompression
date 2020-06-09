import os

import glymur as glymur
import numpy as np
from PIL import Image


class Evaluation:

    def __init__(self) -> None:
        super().__init__()
        self.jpeg200Name = "myfile.jp2"
        self.jpegLossy = "myfile.jpeg"

    def compressionRatio(self, data, imageName):
        jp2 = glymur.Jp2k(self.jpeg200Name, data=data, cratios=[1])
        imPillow = Image.fromarray(data)
        imPillow.save(self.jpegLossy, "JPEG", quality=90)
        # Sizes.
        originalSize = len(data.tostring()) / 1024
        jpeg2000Size = os.stat(self.jpeg200Name).st_size / 1024
        jpegLossySize = os.stat(self.jpegLossy).st_size / 1024
        print('Size of uncompressed {0}: {1} KB'.format(imageName, originalSize))
        jpeg2000CompressionRatio = originalSize / jpeg2000Size
        print('compression ratio of JPEG-2000 encoded {0}: {1}'.format(imageName, jpeg2000CompressionRatio))
        JpegLossyCompressionRatio = originalSize / jpegLossySize
        print('compression ratio of JPEG-lossy encoded {0}: {1}'.format(imageName, JpegLossyCompressionRatio))

        return jpeg2000CompressionRatio, JpegLossyCompressionRatio

    def evaluate(self, filteredData, originalData, inverseFilterFunction=None):
        filteredData = filteredData.astype('uint8')
        originalData = np.abs(originalData).astype('uint8')

        jpeg2000CompressionRatioBefore, JpegLossyCompressionRatioBefore = self.compressionRatio(originalData, "Before")
        jpeg2000CompressionRatioAfter, JpegLossyCompressionRatioAfter = self.compressionRatio(filteredData, "After")

        # Decompress.
        jp2Decoded = glymur.Jp2k(self.jpeg200Name).read()
        if inverseFilterFunction is not None:
            retrivedData = inverseFilterFunction(jp2Decoded)
            retrivedData=retrivedData.astype('uint8')
            pass

        # Compare image data, before and after.
        is_same = (filteredData == jp2Decoded).all()
        print('\nRestored data is identical to original? {:s}\n'.format(str(is_same)))
