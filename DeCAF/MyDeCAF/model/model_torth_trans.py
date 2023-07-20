model.py# DECAF training and predicting model with parallelization
# Created by Renhao Liu and Yu Sun, CIG, WUSTL, 2021

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import skimage
from skimage.metrics import peak_signal_noise_ratio
import cv2
import math
import time
import gc
from absl import flags
import logging

# Model parameters
flags.DEFINE_string("tf_summary_dir", "log", "directory for tf summary log")
flags.DEFINE_enum(
    "positional_encoding_type", "exp_diag", ["exp_diag", "exp", "fourier_fixed_xy"], "positional_encoding_type",)
flags.DEFINE_float("dia_digree", 45, "degrees per each encoding in exp_diag")
flags.DEFINE_enum(
    "mlp_activation", "leaky_relu", ["leaky_relu"], "Activation functions for mlp",)
flags.DEFINE_integer("mlp_layer_num", 10, "number of layers in mlp network")
flags.DEFINE_integer("mlp_kernel_size", 208, "width of mlp")
flags.DEFINE_integer('fourier_encoding_size', 256, "number of rows in fourier matrix")
flags.DEFINE_float("sig_xy", 26.0, "Fourier encoding sig_xy")
flags.DEFINE_float("sig_z", 1.0, "Fourier encoding sig_z")
flags.DEFINE_integer(
    "xy_encoding_num", 6, "number of frequecncies expanded in the spatial dimensions"
)
flags.DEFINE_integer(
    "z_encoding_num", 5, "number of frequecncies expanded in the depth dimension"
)
flags.DEFINE_multi_integer("mlp_skip_layer", [5], "skip layers in the mlp network")
flags.DEFINE_float("output_scale", 5, "neural network out put scale")

# Regularization parameters
flags.DEFINE_enum("regularize_type", "dncnn2d", ["dncnn2d"], "type of the network",)
flags.DEFINE_float("regularize_weight", 0.0, "Weight for regularizer")
flags.DEFINE_float(
    "tv3d_z_reg_weight", 3.079699, "Reg weight scaling for z axis in 3dtv"
)
flags.DEFINE_string(
    "DnCNN_model_path",
    "/export/project/sun.yu/projects/DnCNN/cnn_trained/DnCNN_sigma=5.0/models/final/model",
    "model path of pre-trained DnCNN",
)
flags.DEFINE_float("DnCNN_normalization_min", -0.05, "DnCNN normalization min")
flags.DEFINE_float("DnCNN_normalization_max", 0.05, "DnCNN normalization max")

# Training parameter
flags.DEFINE_integer("start_epoch", 0, "start epoch, useful for continue training")
flags.DEFINE_integer("iters_per_epoch", 1, "num of iters for each resampling")
flags.DEFINE_integer("image_save_epoch", 5000, "number of iteration to save one image")
flags.DEFINE_integer(
    "intermediate_result_save_epoch",
    100,
    "number of iterations to save intermediate result",
)
flags.DEFINE_integer("log_iter", 25, "number of iteration to log to console")
flags.DEFINE_integer("model_save_epoch", 5000, "epoch per intermediate model")
flags.DEFINE_integer(
    "num_measurements_per_batch",
    -1,
    "number of measurements per batch. negative value for all measurements",
)

# Prediction parameters
flags.DEFINE_integer("prediction_batch_size", 1, "Batch size for prediction")

FLAGS = flags.FLAGS

NUM_Z = "nz"
INPUT_CHANNEL = "ic"
OUTPUT_CHANNEL = "oc"
MODEL_SCOPE = "infer_y"
NET_SCOPE = "MLP"
DNCNN_SCOPE = "DnCNN"

# get total number of visible gpus
NUM_GPUS = torch.cuda.device_count()


########################################
###       Tensorboard & Helper       ###
########################################

def record_summary(writer, name, value, step):
    writer.add_scalar(name, value, step)
    writer.flush()


def reshape_image(image):
    if len(image.shape) == 2:
        image_reshaped = image.unsqueeze(0).unsqueeze(-1)
    elif len(image.shape) == 3:
        image_reshaped = image.unsqueeze(-1)
    else:
        image_reshaped = image
    return image_reshaped


def reshape_image_2(image):
    image_reshaped = image.unsqueeze(0).unsqueeze(-1)
    return image_reshaped


def reshape_image_3(image):
    image_reshaped = image.unsqueeze(-1)
    return image_reshaped


def reshape_image_5(image):
    shape = image.shape
    image_reshaped = image.view(-1, shape[2], shape[3], 1)
    return image_reshaped


#################################################
# ***      CLASS OF NEURAL REPRESENTATION     ****
#################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, net_kargs=None, name="model_summary"):
        super(Model, self).__init__()
        # Setup parameters
        self.name = name
        self.tf_summary_dir = "{}/{}".format(FLAGS.tf_summary_dir, name)
        if net_kargs is None:
            self.net_kargs = {
                "skip_layers": FLAGS.mlp_skip_layer,
                "mlp_layer_num": FLAGS.mlp_layer_num,
                "kernel_size": FLAGS.mlp_kernel_size,
                "L_xy": FLAGS.xy_encoding_num,
                "L_z": FLAGS.z_encoding_num,
            }
        else:
            self.net_kargs = net_kargs

    ###########################
    ###     Neural Nets     ###
    ###########################

    def inference(self, coordinates, Hreal, Himag, padding, mask, reuse=False):
        # MLP network
        xhat = self.__neural_repres(
            coordinates, Hreal.shape, **self.net_kargs
        )
        mask = mask.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        Hxhat = self.__forward_op(xhat * mask, Hreal, Himag, padding)
        return Hxhat, xhat

    def __neural_repres(
        self,
        in_node,
        x_shape,
        skip_layers=[],
        mlp_layer_num=10,
        kernel_size=256,
        L_xy=6,
        L_z=5,
    ):
        # positional encoding
        if FLAGS.positional_encoding_type == "exp_diag":
            s = torch.sin(torch.arange(0, 180, FLAGS.dia_digree) * np.pi / 180)[
                :, None
            ]
            c = torch.cos(torch.arange(0, 180, FLAGS.dia_digree) * np.pi / 180)[
                :, None
            ]
            fourier_mapping = torch.cat((s, c), dim=1).T

            xy_freq = torch.matmul(in_node[:, :2], fourier_mapping)

            for l in range(L_xy):
                cur_freq = torch.cat(
                    [
                        torch.sin(2 ** l * np.pi * xy_freq),
                        torch.cos(2 ** l * np.pi * xy_freq),
                    ],
                    dim=-1,
                )
                if l == 0:
                    tot_freq = cur_freq
                else:
                    tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)

                for l in range(L_z):
                    cur_freq = torch.cat(
                        [
                            torch.sin(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)),
                            torch.cos(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)),
                        ],
                        dim=-1,
                    )
                    tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
        elif FLAGS.positional_encoding_type == 'exp':
            for l in range(L_xy):  # fourier feature map
                indicator = torch.tensor([1., 1., 1. if l < L_z else 0.])
                cur_freq = torch.cat([torch.sin(indicator * 2 ** l * np.pi * in_node),
                                      torch.cos(indicator * 2 ** l * np.pi * in_node)], dim=-1)
                if l is 0:
                    tot_freq = cur_freq
                else:
                    tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
        elif FLAGS.positional_encoding_type == 'fourier_fixed_xy':
            torch.manual_seed(10)
            fourier_mapping = torch.normal(0, FLAGS.sig_xy, (FLAGS.fourier_encoding_size, 2)).float()

            xy_freq = torch.matmul(in_node[:, :2], fourier_mapping.T)
            xy_freq = torch.cat([torch.sin(2 * np.pi * xy_freq),
                      torch.cos(2 * np.pi * xy_freq)], dim=-1)

            tot_freq = xy_freq
            for l in range(L_z):
                cur_freq = torch.cat([torch.sin(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)),
                                      torch.cos(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1))], dim=-1)
                tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
        else:
            raise NotImplementedError(FLAGS.positional_encoding_type)
        # input to MLP
        in_node = tot_freq
        # input encoder
        if FLAGS.task_type == "aidt":
            kernel_initializer = None
        elif FLAGS.task_type == "idt":
            kernel_initializer = nn.init.uniform_(-0.05, 0.05)
        elif FLAGS.task_type == "midt":
            kernel_initializer = None
        else:
            raise NotImplementedError

        for layer in range(mlp_layer_num):
            if layer in skip_layers:
                in_node = torch.cat([in_node, tot_freq], -1)

            if FLAGS.mlp_activation == "relu":
                activation = F.relu
            elif FLAGS.mlp_activation == "leaky_relu":
                activation = F.leaky_relu
            elif FLAGS.mlp_activation == "elu":
                activation = F.elu
            elif FLAGS.mlp_activation == "tanh":
                activation = torch.tanh
            in_node = nn.Linear(in_node.size(-1), kernel_size)
            if kernel_initializer is not None:
                kernel_initializer(in_node.weight)
            in_node = activation(in_node)

            if FLAGS.mlp_activation == "leaky_relu":
                in_node = F.leaky_relu(in_node, negative_slope=0.2)

            # final layer
            output = nn.Linear(in_node.size(-1), 2)
            if kernel_initializer is not None:
                kernel_initializer(output.weight)
            output = output / FLAGS.output_scale
        # reshape output to x
        xhat = output.view(x_shape[1], x_shape[2], FLAGS.view_size, FLAGS.view_size, 2)  # [1, Z, X, Y, Real/Imagenary]
        return xhat
    def __forward_op(self,x,Hreal,Himag,padding,    ):
        padded_field = F.pad(x, padding)
        padded_phase = padded_field[:, :, :, :, 0]
        padded_absorption = padded_field[:, :, :, :, 1]
        transferred_field = torch.fft.ifft2(
            torch.mul(Hreal, torch.fft.fft2(padded_phase.to(torch.complex64)))
            + torch.mul(Himag, torch.fft.fft2(padded_absorption.to(torch.complex64)))
        )
        Hxhat = torch.sum(torch.real(transferred_field), dim=(1, 2))
        return Hxhat



    def restore(self, model_path):
        self.load_state_dict(torch.load(model_path))



