import time
import sys

import cv2
from helper import utilty as util

class Timer(object):
    def __init__(self, name):
        self.name = name
        self.start = 0
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, *a, **kw):
        print('%s took %.3f seconds' % (self.name, time.time() - self.start))

def do_util(src, dest):
    with Timer('util: load'):
        inp = util.load_image(src)
    with Timer('util: resize'):
        resized = util.resize_image_by_pil(inp, 2)
    with Timer('util: extract Y'):
        only_y = util.convert_rgb_to_y(inp)
    only_y = util.convert_rgb_to_y(resized) # simulate upscale
    with Timer('util: rgb => YCbCr'):
        scaled_ycbcr_image = util.convert_rgb_to_ycbcr(resized)
    with Timer('util: Y + YCbCr -> rgb'):
        image = util.convert_y_and_cbcr_to_rgb(only_y, scaled_ycbcr_image[:, :, 1:3])
    with Timer('util: save'):
        util.save_image(dest, image)

def do_cv(src, dest):
    with Timer('cv2: load'):
        inp = cv2.imread(src)
    with Timer('cv2: resize'):
        resized = cv2.resize(inp, (inp.shape[1] * 2, inp.shape[0] * 2))
    with Timer('cv2: extract Y'):
        ycbcr = cv2.cvtColor(inp, cv2.COLOR_RGB2YCR_CB)
        only_Cb, only_Cr, only_y = cv2.split(ycbcr)
    # simulate upscale
    ycbcr = cv2.cvtColor(resized, cv2.COLOR_RGB2YCR_CB)
    only_Cb, only_Cr, only_y = cv2.split(ycbcr)

    with Timer('cv2: rgb => YCbCr'):
        scaled_ycbcr_image = cv2.cvtColor(resized, cv2.COLOR_RGB2YCR_CB)
    with Timer('cv2: Y + YCbCr -> rgb'):
        new_Cb, new_Cr, new_y = cv2.split(scaled_ycbcr_image)
        image = cv2.merge((new_Cb, new_Cr, only_y ))
        image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2RGB)
    with Timer('cv2: save'):
        cv2.imwrite(dest, image)


if __name__ == '__main__':
    src = sys.argv[1:]
    if not src:
        sys.exit('Usage: %s test-image.png [test-image1.png]' % sys.argv[0])
    for img in src:
        with Timer('util convert'):
            do_util(img, 'b_util.png')
        print('-' * 10)
        with Timer('cv2 convert'):
            do_cv(img, 'b_cv.png')
