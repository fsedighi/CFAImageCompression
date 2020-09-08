import cv2
import numpy as np



class RGB2CFAUtils:

    def __init__(self) -> None:
        super().__init__()

    def rgb2CFA(self, image, show=False,res=0):
        h, w, c = image.shape
        if res==0:
            cfa = np.zeros((h, w), dtype=np.uint8)
        else:
            cfa = np.zeros((h, w), dtype=np.uint16)
        red = image[range(1, h, 2), :, :][:, range(0, w, 2), :][:, :, 0]
        green1 = image[range(0, h, 2), :, :][:, range(0, w, 2), :][:, :, 1]
        green2 = image[range(1, h, 2), :, :][:, range(1, w, 2), :][:, :, 1]
        blue = image[range(0, h, 2), :, :][:, range(1, w, 2), :][:, :, 2]

        cfa[np.ix_(range(1, h, 2), range(0, w, 2))] = red
        cfa[np.ix_(range(0, h, 2), range(0, w, 2))] = green1
        cfa[np.ix_(range(1, h, 2), range(1, w, 2))] = green2
        cfa[np.ix_(range(0, h, 2), range(1, w, 2))] = blue

        if show:
            cv2.imshow('red', red)
            cv2.imshow('blue', blue)
            cv2.imshow('green1', green1)
            cv2.imshow('green2', green2)
            cv2.imshow('CFA', cfa)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return cfa


