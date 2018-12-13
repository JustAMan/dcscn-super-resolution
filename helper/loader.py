"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2

functions for loading/converting data
"""

import logging
import os
import random
import tempfile

import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter, gaussian_filter1d

from helper import utilty as util

INPUT_IMAGE_DIR = "input"
INTERPOLATED_IMAGE_DIR = "interpolated"
TRUE_IMAGE_DIR = "true"

INPUT_SUFFIX = "-input"
TRUE_SUFFIX = "-true"


class InputImageMaker:
    def make_input_image(self, file_path, true_image, scale=1,
                         resampling_method='bicubic', print_console=True):
        raise NotImplementedError("Abstract method")


class BlurryJpegifiedInputImageMaker(InputImageMaker):
    def __init__(self, hblur_max=70, vblur_max=70, jpegify=(40, 110), resampling_method='bicubic'):
        InputImageMaker.__init__(self)
        self.hblur_max = hblur_max
        self.vblur_max = vblur_max
        self.jpegify = jpegify
        self.resampling_method = resampling_method

    def make_input_image(self, file_path, true_image, scale=1, print_console=True):
        if true_image is not None:
            assert len(true_image.shape) == 3 and true_image.shape[2] == 3

        fname, fext = os.path.splitext(file_path)
        input_image = None
        if fname.lower().endswith(TRUE_SUFFIX):
            target = fname[:-len(TRUE_SUFFIX)] + INPUT_SUFFIX + fext
            head, tail = os.path.split(target)
            target = os.path.join(head, 'input', tail)
            if os.path.exists(target):
                input_image = util.set_image_alignment(util.load_image(target, print_console=print_console), scale)

        if input_image is None and true_image is not None:
            input_image = util.resize_image_by_pil(true_image, 1.0 / scale, resampling_method=self.resampling_method)

        hblur_radius = random.randrange(0, self.hblur_max) / 100.
        vblur_radius = random.randrange(0, self.vblur_max) / 100.
        qua = random.randrange(self.jpegify[0], self.jpegify[1])

        if vblur_radius > 0:
            input_image = gaussian_filter1d(input_image, sigma=vblur_radius, axis=0)
        if hblur_radius > 0:
            input_image = gaussian_filter1d(input_image, sigma=hblur_radius, axis=1)
        if qua < 100:
            tmpname = tempfile.mktemp(suffix='.jpg')
            util.save_image(tmpname, input_image, qua, print_console=False)
            input_image = util.load_image(tmpname, print_console=False)
            os.unlink(tmpname)

        return input_image


def build_input_image(image, scale=1, alignment=0):
    if alignment > 1:
        image = util.set_image_alignment(image, alignment)

    if scale != 1:
        image = util.resize_image_by_pil(image, 1.0 / scale)

    return image


class DynamicDataSets:
    def __init__(self, scale, batch_image_size, channels=1, resampling_method="bicubic"):
        self.scale = scale
        self.batch_image_size = batch_image_size
        self.channels = channels
        self.resampling_method = resampling_method

        self.filenames = []
        self.count = 0
        self.batch_index = None

    def set_data_dir(self, data_dir):
        self.filenames = util.get_files_in_directory(data_dir)
        self.count = len(self.filenames)
        if self.count <= 0:
            logging.error("Data Directory is empty.")
            exit(-1)

    def init_batch_index(self):
        self.batch_index = random.sample(range(0, self.count), self.count)
        self.index = 0

    def get_next_image_no(self):

        if self.index >= self.count:
            self.init_batch_index()

        image_no = self.batch_index[self.index]
        self.index += 1
        return image_no

    def load_batch_image(self, max_value):
        raise NotImplementedError

    def load_random_patch(self, filename):
        raise NotImplementedError


class DynamicDataSetsWithInput(DynamicDataSets):
    def __init__(self, scale, batch_image_size, image_maker: InputImageMaker, channels=1, resampling_method="bicubic"):
        DynamicDataSets.__init__(self, scale, batch_image_size, channels, resampling_method)
        self.image_maker = image_maker

    def load_batch_image(self, max_value):
        image = None
        file_path = None
        while image is None:
            file_path = self.filenames[self.get_next_image_no()]
            image = self.load_random_patch(file_path)

        input_image = self.image_maker.make_input_image(file_path, image, scale=self.scale,
                                                        print_console=False)

        flip = random.randrange(0, 4)
        if flip == 1 or flip == 3:
            input_image = np.flipud(input_image)
            image = np.flipud(image)
        if flip == 2 or flip == 3:
            input_image = np.fliplr(input_image)
            image = np.fliplr(image)

        rot90 = random.randrange(0, 2)
        if rot90 == 1:
            input_image = np.rot90(input_image)
            image = np.rot90(image)

        input_image = util.convert_rgb_to_y(input_image)
        input_bicubic_image = util.resize_image_by_pil(input_image, self.scale)

        if max_value != 255:
            scale = max_value / 255.0
            input_image = np.multiply(input_image, scale)
            input_bicubic_image = np.multiply(input_bicubic_image, scale)
            image = np.multiply(image, scale)

        image = util.convert_rgb_to_y(image)

        return input_image, input_bicubic_image, image

    def load_random_patch(self, filename):

        image = util.load_image(filename, print_console=False)
        height, width = image.shape[0:2]

        load_batch_size = self.batch_image_size * self.scale

        if height < load_batch_size or width < load_batch_size:
            print("Error: %s should have more than %d x %d size." % (filename, load_batch_size, load_batch_size))
            return None, None, None

        if height == load_batch_size:
            y = 0
        else:
            y = random.randrange(height - load_batch_size)

        if width == load_batch_size:
            x = 0
        else:
            x = random.randrange(width - load_batch_size)
        x -= x % self.scale
        y -= y % self.scale
        image = image[y:y + load_batch_size, x:x + load_batch_size, :]
        image = build_input_image(image)

        return image

