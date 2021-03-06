"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2

functions for sharing arguments and their default values
"""

import sys

import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Model (network) Parameters
flags.DEFINE_integer("scale", 2, "Scale factor for Super Resolution (should be 2 or more)")
flags.DEFINE_integer("layers", 12, "Number of layers of feature xxtraction CNNs")
flags.DEFINE_integer("filters", 196, "Number of filters of first feature-extraction CNNs")
flags.DEFINE_integer("min_filters", 48, "Number of filters of last feature-extraction CNNs")
flags.DEFINE_float("filters_decay_gamma", 1.5,
                   "Number of CNN filters are decayed from [filters] to [min_filters] by this gamma")
flags.DEFINE_boolean("use_nin", True, "Use Network In Network")
flags.DEFINE_integer("nin_filters", 64, "Number of CNN filters in A1 at Reconstruction network")
flags.DEFINE_integer("nin_filters2", 32, "Number of CNN filters in B1 and B2 at Reconstruction net.")
flags.DEFINE_integer("cnn_size", 3, "Size of CNN filters")
flags.DEFINE_integer("reconstruct_layers", 1, "Number of Reconstruct CNN Layers. (can be 0.)")
flags.DEFINE_integer("reconstruct_filters", 32, "Number of Reconstruct CNN Filters")
flags.DEFINE_float("dropout_rate", 0.8, "Output nodes should be kept by this probability. If 1, don't use dropout.")
flags.DEFINE_string("activator", "prelu", "Activator can be [relu, leaky_relu, prelu, sigmoid, tanh, selu]")
flags.DEFINE_boolean("pixel_shuffler", True, "Use Pixel Shuffler instead of transposed CNN")
flags.DEFINE_integer("pixel_shuffler_filters", 0,
                     "Num of Pixel Shuffler output channels. 0 means use same channels as input.")
flags.DEFINE_integer("self_ensemble", 1, "Number of using self ensemble method. [1 - 8]")
flags.DEFINE_boolean("batch_norm", False, "use batch normalization after each CNNs")

# Training Parameters
flags.DEFINE_boolean("bicubic_init", True, "make bicubic interpolation values as initial input for x2")
flags.DEFINE_float("clipping_norm", 5, "Norm for gradient clipping. If it's <= 0 we don't use gradient clipping.")
flags.DEFINE_string("initializer", "he", "Initializer for weights can be [uniform, stddev, xavier, he, identity, zero]")
flags.DEFINE_float("weight_dev", 0.01, "Initial weight stddev (won't be used when you use he or xavier initializer)")
flags.DEFINE_float("l2_decay", 0.0001, "l2_decay")
flags.DEFINE_string("optimizer", "adam", "Optimizer can be [gd, momentum, adadelta, adagrad, adam, rmsprop]")
flags.DEFINE_float("beta1", 0.9, "Beta1 for adam optimizer")
flags.DEFINE_float("beta2", 0.999, "Beta2 for adam optimizer")
flags.DEFINE_float("epsilon", 1e-8, "epsilon for adam optimizer")
flags.DEFINE_float("momentum", 0.9, "Momentum for momentum optimizer and rmsprop optimizer")
flags.DEFINE_integer("batch_num", 1, "Number of mini-batch images for training")
flags.DEFINE_integer("batch_image_size", 48, "Image size for mini-batch")
flags.DEFINE_integer("stride_size", 0, "Stride size for mini-batch. If it is 0, use half of batch_image_size")
flags.DEFINE_integer("training_images", 24000, "Number of training on each epoch")
flags.DEFINE_boolean("use_l1_loss", False, "Use L1 Error as loss function instead of MSE Error.")

# Learning Rate Control for Training
flags.DEFINE_float("initial_lr", 0.002, "Initial learning rate")
flags.DEFINE_float("lr_decay", 0.75, "Learning rate decay rate")
flags.DEFINE_integer("lr_decay_epoch", 4, "After this epochs are completed, learning rate will be decayed by lr_decay.")
flags.DEFINE_float("end_lr", 2e-5, "Training end learning rate. If the current learning rate gets lower than this"
                                   "value, then training will be finished.")

# Dataset or Others
flags.DEFINE_string("dataset", "bsd200", "Training dataset dir. [yang91, general100, bsd200, other]")
flags.DEFINE_string("test_dataset", "set5", "Directory for test dataset [set5, set14, bsd100, urban100, all]")
flags.DEFINE_integer("tests", 1, "Number of training sets")
flags.DEFINE_boolean("do_benchmark", False, "Evaluate the performance for set5, set14 and bsd100 after the training.")

# Image Processing
flags.DEFINE_integer("hblur_max", 50, "Horizontal blur of input image for training "
                                      "(0 means no blur, 100 means one pixel blur)")
flags.DEFINE_integer("vblur_max", 50, "Vertical blur of input image for training "
                                      "(0 means no blur, 100 means one pixel blur)")
flags.DEFINE_integer("jpegify_min", 60, "Minimal quality of JPEG encoding of the input image for training (0..100)")
flags.DEFINE_integer("jpegify_max", 120, "Maximal quality of JPEG encoding of the input image for training "
                                         "(can be greater than 100 that means no JPEG encoding with some probability)")
flags.DEFINE_float("patch_scale_max", 1.5, "Maximum scale of a training patch additional to the scale factor")

flags.DEFINE_float("max_value", 255, "For normalize image pixel value")
flags.DEFINE_integer("channels", 1, "Number of image channels used. Now it should be 1. using only Y from YCbCr.")
flags.DEFINE_integer("psnr_calc_border_size", -1,
                     "Cropping border size for calculating PSNR. if < 0, use 2 + scale for default.")

# Environment (all directory name should not contain '/' after )
flags.DEFINE_string("checkpoint_dir", "models", "Directory for checkpoints")
flags.DEFINE_string("graph_dir", "graphs", "Directory for graphs")
flags.DEFINE_string("data_dir", "data", "Directory for original images")
flags.DEFINE_string("batch_dir", "batch_data", "Directory for training batch images")
flags.DEFINE_string("output_dir", "output", "Directory for output test images")
flags.DEFINE_string("tf_log_dir", "tf_log", "Directory for tensorboard log")
flags.DEFINE_string("log_filename", "log.txt", "log filename")
flags.DEFINE_string("model_name", "", "model name for save files and tensorboard log")
flags.DEFINE_string("load_model_name", "", "Filename of model loading before start [filename or 'default']")

# Debugging or Logging
flags.DEFINE_boolean("initialize_tf_log", True, "Clear all tensorboard log before start")
flags.DEFINE_boolean("enable_log", True, "Enables tensorboard-log. Save loss.")
flags.DEFINE_boolean("save_weights", True, "Save weights and biases/gradients")
flags.DEFINE_boolean("save_images", False, "Save CNN weights as images")
flags.DEFINE_integer("save_images_num", 20, "Number of CNN images saved")
flags.DEFINE_boolean("save_meta_data", False, "")
flags.DEFINE_integer("gpu_device_id", 0, "Device ID of GPUs which will be used to compute.")


def get():
    sys.stderr.write("Python Interpreter version: %s\n" % sys.version[:3])
    sys.stderr.write("tensorflow version: %s\n" % tf.__version__)
    sys.stderr.write("numpy version: %s\n" % np.__version__)

    # check which library you are using
    # np.show_config()
    return FLAGS
