import os
import time
import glob
import sys
import builtins

orig_print = builtins.print
prev_flush = time.time()
def print_err(*args, **kw):
    global prev_flush
    if len(args) > 1 or kw:
        sys.stderr.write('[SR] called print() with more than 1 arg!!\n')
        return orig_print(*args, **kw)
    sys.stderr.write('%s\n' % args[0])
    if time.time() - prev_flush > 1:
        sys.stderr.flush()
        prev_flush = time.time()
builtins.print = print_err

from scipy.ndimage.filters import gaussian_filter
import numpy as np

import tensorflow as tf

import DCSCN
import cv2
from helper import args
from helper import utilty as util

args.flags.DEFINE_string("file", "image.jpg", "Target filename")
args.flags.DEFINE_string("file_glob", "", "Target filenames pattern")
args.flags.DEFINE_bool("pipe", False, "Communicate via pipes")
args.flags.DEFINE_string("img_size", "", "Image size that is passed via pipe, e.g. 720x540")
FLAGS = args.get()

def parse_stdin(width, height):
    frame_size = width * height * 4
    while True:
        data = sys.stdin.buffer.read(frame_size)
        if not data: # end of input
            break
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.cvtColor(arr.reshape(height, width, 4)[:, :, 1:4], cv2.COLOR_BGR2RGB)
        yield img


ALPHA_BUF = None
def pack_image_to_stdout(img):
    global ALPHA_BUF
    arr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if ALPHA_BUF is not None:
        assert arr.shape[:-1] == ALPHA_BUF.shape[:-1]
    else:
        ALPHA_BUF = np.ones((arr.shape[0], arr.shape[1], 4), dtype=np.uint8) * 255
    ALPHA_BUF[:, :, 1:4] = arr
    sys.stdout.buffer.write(ALPHA_BUF.tobytes())


class Upscaler(DCSCN.SuperResolution):
    def upscale(self, org_image):
        assert len(org_image.shape) >= 3 and org_image.shape[2] == 3 and self.channels == 1

        input_y_image = util.convert_rgb_to_y(org_image)

        big_blurry_input_image = util.resize_image_by_pil(org_image, self.scale)
        big_blurry_input_ycbcr_image = util.convert_rgb_to_ycbcr(big_blurry_input_image)
        bbi_input_y_image, bbi_input_cbcr_image = util.convert_ycbcr_to_y_cbcr(big_blurry_input_ycbcr_image)

        output_y_image = self.do(input_y_image, bbi_input_y_image)

        output_image = util.convert_y_and_cbcr_to_rgb(output_y_image, bbi_input_cbcr_image)
        return output_image

    def do_for_file(self, file_path, output_folder="output"):
        org_image = cv2.imread(file_path)
        output_image = self.upscale(org_image)
        target_path = os.path.basename(file_path)
        cv2.imwrite(os.path.join(output_folder, target_path), output_image)


def upscale(model, fname, output):
    start = time.time()
    model.do_for_file(fname, output)
    print('upscaling took: %.3f seconds' % (time.time() - start))


def main(_):
    model = Upscaler(FLAGS, model_name=FLAGS.model_name)
    model.build_graph()
    model.build_optimizer()
    model.build_summary_saver()

    model.init_all_variables()
    model.load_model()

    if FLAGS.pipe:
        try:
            width, height = [int(x) for x in FLAGS.img_size.lower().split('x')]
        except ValueError:
            sys.exit('Invalid --img_size passed: %s' % FLAGS.img_size)
        for idx, img in enumerate(parse_stdin(width, height)):
            start = time.time()
            upped = model.upscale(img)
            pack_image_to_stdout(upped)
            print_err('[SR] img: %d, upscaling took: %.3f' % (idx, time.time() - start))

    else:
        lst = None
        if FLAGS.file_glob:
            lst = glob.glob(FLAGS.file_glob)
        if not lst:
            lst = [FLAGS.file]
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        for fname in lst:
            upscale(model, fname, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
