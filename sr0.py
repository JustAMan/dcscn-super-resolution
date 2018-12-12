import os
import time
import glob

from scipy.ndimage.filters import gaussian_filter
import numpy as np

import tensorflow as tf

import DCSCN
import cv2
from helper import args
from helper import utilty as util

args.flags.DEFINE_string("file", "image.jpg", "Target filename")
args.flags.DEFINE_string("file_glob", "", "Target filenames pattern")
FLAGS = args.get()


class Upscaler(DCSCN.SuperResolution):
    def do_for_file(self, file_path, output_folder="output"):
        org_image = cv2.imread(file_path)
        assert len(org_image.shape) >= 3 and org_image.shape[2] == 3 and self.channels == 1

        input_y_image = util.convert_rgb_to_y(org_image)

        big_blurry_input_image = util.resize_image_by_pil(org_image, self.scale)
        big_blurry_input_ycbcr_image = util.convert_rgb_to_ycbcr(big_blurry_input_image)
        bbi_input_y_image, bbi_input_cbcr_image = util.convert_ycbcr_to_y_cbcr(big_blurry_input_ycbcr_image)

        output_y_image = self.do(input_y_image, bbi_input_y_image)

        output_image = util.convert_y_and_cbcr_to_rgb(output_y_image, bbi_input_cbcr_image)

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
