# DECAF training and predicting model with parallelization
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




class Loss(nn.Module):
    def __init__(self, tower_idx=None, Hreal=None, Himag=None):
        super(Loss, self).__init__()
        self.tower_idx=tower_idx
        self.Hreal = Hreal
        self.Himag = Himag

        # Setup parameters

    ##############################
    ###     Loss Functions     ###
    ##############################

    def forward(self, FLAGS, Hxhat, xhat, y, tower_idx=0, reuse=False):
        # get input coordinates & measurements & padding
        # x = self.Xs[tower_idx, ...]
        # y = self.Ys[tower_idx, ...]
        # padding = self.Ps[tower_idx, ...]
        # mask = self.Ms[tower_idx, ...]

        # inference

        # data fidelity
        if FLAGS.loss == "l1":
            mse = torch.mean(torch.abs(Hxhat - y))
        elif FLAGS.loss == "l2":
            mse = torch.mean(torch.square(Hxhat - y)) / 2
        else:
            raise NotImplementedError
        # regularizer

        # if FLAGS.regularize_type == "dncnn2d":
        #     xhat_trans = torch.transpose(
        #         torch.squeeze(xhat), 3, 0
        #     )  # [1, Z, X, Y, Real/Imagenary]
        #     xhat_concat = torch.cat([xhat_trans[0, ...], xhat_trans[1, ...]], 0)
        #     xhat_expand = xhat_concat.unsqueeze(3)
        #     phase_regularize_value = self.__dncnn_2d(FLAGS, xhat_expand.to('cpu'), reuse=reuse)
        #     absorption_regularize_value = torch.tensor(0.0)
        # else:
        #     raise NotImplementedError

        phase_regularize_value=0.5
        absorption_regularize_value=0.5
        if FLAGS.tv3d_z_reg_weight != 0:
            tv_z = self.__total_variation_z(xhat[..., 0])
            tv_z += self.__total_variation_z(xhat[..., 1])
        else:
            tv_z = torch.tensor(0.0)

        # final loss
        loss = (
            mse
            + FLAGS.regularize_weight
            #* (absorption_regularize_value + phase_regularize_value)
            + FLAGS.tv3d_z_reg_weight * tv_z
        )

        return (
            loss,
            mse,
            phase_regularize_value,
            absorption_regularize_value,
            # xhat,
            # Hxhat,
            # y,
        )

    def __total_variation_2d(self, images):
        pixel_dif2 = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
        pixel_dif3 = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
        total_var = torch.sum(pixel_dif2) + torch.sum(pixel_dif3)
        return total_var

    def __total_variation_z(self, images):
        """
        Normalized total variation 3d
        :param images: Images should have 4 dims: batch_size, z, x, y
        :return:
        """
        pixel_dif1 = torch.abs(images[:, 1:, :, :] - images[:, :-1, :, :])
        total_var = torch.sum(pixel_dif1)
        return total_var
    def __dncnn_inference(
        self,
        input,
        reuse,
        output_channel=1,
        layer_num=10,
        filter_size=3,
        feature_root=64,
    ):
        # input layer
        with torch.no_grad():
            in_node = nn.Conv2d(input.size(1), feature_root, filter_size, padding=filter_size//2)
            in_node = F.relu(in_node)
            # composite convolutional layers
            for layer in range(2, layer_num):
                in_node = nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size//2, bias=False)
                in_node = F.relu(nn.BatchNorm2d(feature_root)(in_node))
            # output layer and residual learning
            in_node = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size//2)
            output = input - in_node
        return output




    def __dncnn_2d(self, FLAGS, images,reuse=True):  # [N, H, W, C]
        """
        DnCNN as 2.5 dimensional denoiser based on l-2 norm
        """
        a_min = FLAGS.DnCNN_normalization_min
        a_max = FLAGS.DnCNN_normalization_max
        normalized = (images - a_min) / (a_max - a_min)
        denoised = self.__dncnn_inference(torch.clamp(normalized, 0, 1),reuse)
        denormalized = denoised * (a_max - a_min) + a_min
        dncnn_res = torch.sum(denormalized**2)
        return dncnn_res


class dncnn_2d(nn.Module):
    def __int__(self,FLAGS,output_channel=1,layer_num=10,filter_size=3,feature_root=64,):
        super(dncnn_2d,self).__init__()
        self.input_conv=nn.Conv2d(input.size(1), feature_root, filter_size, padding=filter_size // 2)
        self.convs=nn.ModuleList([
            nn.Linear(FLAGS.mlp_kernel_size, FLAGS.mlp_kernel_size) for i in range(FLAGS.mlp_skip_layer[0])
        ])
        self.output_conv=nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size // 2)

        in_node = nn.Conv2d(input.size(1), feature_root, filter_size, padding=filter_size // 2)
        in_node = F.relu(in_node)
        # composite convolutional layers
        for layer in range(2, layer_num):
            in_node = nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size // 2, bias=False)
            in_node = F.relu(nn.BatchNorm2d(feature_root)(in_node))
        # output layer and residual learning
        in_node = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size // 2)

    def forward(self,FLAGS, images,reuse=True):
        a_min = FLAGS.DnCNN_normalization_min
        a_max = FLAGS.DnCNN_normalization_max
        normalized = (images - a_min) / (a_max - a_min)
        denoised = self.__dncnn_inference(torch.clamp(normalized, 0, 1),reuse)
        denormalized = denoised * (a_max - a_min) + a_min
        dncnn_res = torch.sum(denormalized**2)
        return dncnn_res

        return 0
    def __dncnn_inference(self,input):
        x=self.input_conv(input)
        for f in self.convs:
            x=f(x)
        output=self.output_conv(x)

        return output