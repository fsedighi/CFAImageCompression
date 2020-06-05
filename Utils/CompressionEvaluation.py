import os

import glymur as glymur
import numpy as np
from PIL import Image


class CompressionEvaluation:

    def __init__(self) -> None:
        super().__init__()
        self.jpeg200Name = "myfile.jp2"
        self.jpegLossy = "myfile.jpeg"

    def evaluate(self, data_image, imageName):
        # Compress image data to a sequence of bytes.
        # data_image = data_image + np.abs(np.min(data_image))
        # data_image = data_image / np.max(data_image)  # normalize the data to 0 - 1
        # data_image = 255 * data_image  # Now scale by 255
        data_image = np.abs(data_image).astype('uint8')

        jp2 = glymur.Jp2k(self.jpeg200Name, data=data_image, cratios=[1])
        imPillow = Image.fromarray(data_image)
        imPillow.save(self.jpegLossy, "JPEG", quality=95)
        # Sizes.
        originalSize = len(data_image.tostring()) / 1024
        jpeg2000Size = os.stat(self.jpeg200Name).st_size / 1024
        jpegLossySize = os.stat(self.jpegLossy).st_size / 1024
        print('Size of uncompressed {0}: {1} KB'.format(imageName, originalSize))
        print('compression ratio of JPEG-2000 encoded {0}: {1} KB'.format(imageName, originalSize / jpeg2000Size))
        print('compression ratio of JPEG-lossy encoded {0}: {1} KB'.format(imageName, originalSize / jpegLossySize))
        # Decompress.
        jp2Decoded = glymur.Jp2k(self.jpeg200Name).read()

        # Compare image data, before and after.
        is_same = (data_image == jp2Decoded).all()
        print('\nRestored data is identical to original? {:s}\n'.format(str(is_same)))
