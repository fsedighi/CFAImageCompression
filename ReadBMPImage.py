
def load_bmp(bmp_image):
    with open(bmp_image, 'rb') as f:
        data = bytearray(f.read())
    # convert byte array to 2 dimension array with integer values
    bayer = "2 dimension array with integer values"
    return bayer
