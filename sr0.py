"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Author: Jin Yamanaka
Github: https://github.com/jiny2001/dcscn-image-super-resolution
Ver: 2.0

Apply Super Resolution for image file.

--file=[your image filename]: will generate HR images.
see output/[model_name]/ for checking result images.

Also you must put same model args as you trained.

For ex, if you trained like below,
> python train.py --scale=3

Then you must run sr.py like below.
> python sr.py --scale=3 --file=your_image_file_path


If you trained like below,
> python train.py --dataset=bsd200 --layers=8 --filters=96 --training_images=30000

Then you must run sr.py like below.
> python sr.py --layers=8 --filters=96 --file=your_image_file_path

"""

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

def cv_convert_rgb_to_y(image):
    ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    only_Cb, only_Cr, only_y = cv2.split(ycbcr)
    return only_y


def do_cv(src, dest):
        inp = cv2.imread(src)

        resized = cv2.resize(inp, (inp.shape[1] * 2, inp.shape[0] * 2))

        ycbcr = cv2.cvtColor(resized, cv2.COLOR_RGB2YCR_CB)
        only_Cb, only_Cr, only_y = cv2.split(ycbcr)

        scaled_ycbcr_image = cv2.cvtColor(resized, cv2.COLOR_RGB2YCR_CB)

        new_Cb, new_Cr, new_y = cv2.split(scaled_ycbcr_image)
        image = cv2.merge((new_Cb, new_Cr, only_y ))
        image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2RGB)

        cv2.imwrite(dest, image)


class Upscaler(DCSCN.SuperResolution):
    def do_for_file(self, file_path, output_folder="output"):
        org_image = cv2.imread(file_path)                   #org_image = util.load_image(file_path)
        assert len(org_image.shape) >= 3 and org_image.shape[2] == 3 and self.channels == 1

        input_ycbcr_image = cv2.cvtColor(org_image, cv2.COLOR_RGB2YCR_CB)
        only_Cb, only_Cr, input_y_image = cv2.split(input_ycbcr_image)

        big_blurry_input_image = cv2.resize(org_image, (org_image.shape[1] * self.scale, org_image.shape[0] * self.scale))
        big_blurry_input_ycbcr_image = cv2.cvtColor(big_blurry_input_image, cv2.COLOR_RGB2YCR_CB)
        bbi_only_Cb, bbi_only_Cr, bbi_input_y_image = cv2.split(big_blurry_input_ycbcr_image)

        output_y_image = self.do(input_y_image, bbi_input_y_image)

        output_ycbcr_image = cv2.merge((bbi_only_Cb, bbi_only_Cr, output_y_image))
        output_image = cv2.cvtColor(output_ycbcr_image, cv2.COLOR_YCR_CB2RGB)

        target_path = os.path.basename(file_path)
        cv2.imwrite(os.path.join(output_folder, target_path), output_image)      #util.save_image(os.path.join(output_folder, target_path), image)

    def do(self, y_input_image, bicubic_y_input_image=None):

        h, w = y_input_image.shape[:2]
        ch = y_input_image.shape[2] if len(y_input_image.shape) > 2 else 1

        assert bicubic_y_input_image is not None

        if self.max_value != 255.0:
            y_input_image *= self.max_value / 255.0
            bicubic_y_input_image *= self.max_value / 255.0

        assert self.self_ensemble == 1

        y = self.sess.run(self.y_, feed_dict={self.x: y_input_image.reshape(1, h, w, ch),
                                              self.x2: bicubic_y_input_image.reshape(1, self.scale * h,
                                                                                     self.scale * w, ch),
                                              self.dropout: 1.0, self.is_training: 0})
        output_image = y[0,:,:,0]

        if self.max_value != 255.0:
            output_image *= 255.0 / self.max_value

        return output_image.astype(np.uint8)


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
