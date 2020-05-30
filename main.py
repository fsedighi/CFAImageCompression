from ReadBMPImage import load_bmp
from ORCT1 import compute_orct1


image = open('image.bmp')
bayer = load_bmp(image)
compute_orct1(bayer)
