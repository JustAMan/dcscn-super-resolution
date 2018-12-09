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

import tensorflow as tf

import DCSCN
from helper import args
from helper import utilty as util

args.flags.DEFINE_string("file", "image.jpg", "Target filename")
args.flags.DEFINE_string("file_glob", "", "Target filenames pattern")
FLAGS = args.get()


class Upscaler(DCSCN.SuperResolution):
    def do_for_file(self, file_path, output_folder="output", blur=None):
        org_image = util.load_image(file_path)
        assert len(org_image.shape) >= 3 and org_image.shape[2] == 3 and self.channels == 1

        input_y_image = util.convert_rgb_to_y(org_image)
        if blur is not None:
            input_y_image = gaussian_filter(input_y_image, sigma=blur)
        output_y_image = self.do(input_y_image)
        scaled_ycbcr_image = util.convert_rgb_to_ycbcr(
            util.resize_image_by_pil(org_image, self.scale, self.resampling_method))
        image = util.convert_y_and_cbcr_to_rgb(output_y_image, scaled_ycbcr_image[:, :, 1:3])

        target_path = os.path.basename(file_path)
        if blur is not None:
            head, ext = os.path.splitext(target_path)
            target_path = '%s-blur-%.2f%s' % (head, blur, ext)
        util.save_image(os.path.join(output_folder, target_path), image)


def upscale(model, fname, output):
    start = time.time()
    model.do_for_file(fname, output)
    #for blur in (0.2, 0.4, 0.6, 0.8):
    #    model.do_for_file(fname, output, blur=blur)
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
    for fname in lst:
        upscale(model, fname, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
