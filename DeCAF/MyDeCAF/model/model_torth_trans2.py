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
            # elif FLAGS.positional_encoding_type == 'exp':
            #     for l in range(L_xy):  # fourier feature map
            #         indicator = torch.tensor([1., 1., 1. if l < L_z else 0.])
            #         cur_freq = torch.cat([torch.sin(indicator * 2 ** l * np.pi * in_node),
            #                               torch.cos(indicator * 2 ** l * np.pi * in_node)], dim=-1)
            #         if l is 0:
            #             tot_freq = cur_freq
            #         else:
            #             tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
            # elif FLAGS.positional_encoding_type == 'fourier_fixed_xy':
            #     torch.manual_seed(10)
            #     fourier_mapping = torch.normal(0, FLAGS.sig_xy, (FLAGS.fourier_encoding_size, 2)).float()
            #
            #     xy_freq = torch.matmul(in_node[:, :2], fourier_mapping.T)
            #     xy_freq = torch.cat([torch.sin(2 * np.pi * xy_freq),
            #               torch.cos(2 * np.pi * xy_freq)], dim=-1)
            #
            #     tot_freq = xy_freq
            #     for l in range(L_z):
            #         cur_freq = torch.cat([torch.sin(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)),
            #                               torch.cos(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1))], dim=-1)
            #         tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
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

    def save(self, directory, epoch=None, train_provider=None):
        if epoch is not None:
            directory = os.path.join(directory, "{}_model/".format(epoch))
        else:
            directory = os.path.join(directory, "latest/".format(epoch))
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, "model")
        if train_provider is not None:
            train_provider.save(directory)
        torch.save(self.state_dict(), path)
        print("saved to {}".format(path))
        return path

    def restore(self, model_path):
        self.load_state_dict(torch.load(model_path))

    ##############################
    ###     Loss Functions     ###
    ##############################

    def __tower_loss(self, tower_idx, Hreal, Himag, reuse=False):
        # get input coordinates & measurements & padding
        x = self.Xs[tower_idx, ...]
        y = self.Ys[tower_idx, ...]
        padding = self.Ps[tower_idx, ...]
        mask = self.Ms[tower_idx, ...]

        # inference
        Hxhat, xhat = self.inference(x, Hreal, Himag, padding, mask, reuse=reuse)
        # data fidelity
        if FLAGS.loss == "l1":
            mse = torch.mean(torch.abs(Hxhat - y))
        elif FLAGS.loss == "l2":
            mse = torch.mean(torch.square(Hxhat - y)) / 2
        else:
            raise NotImplementedError
        # regularizer
        if FLAGS.regularize_type == "dncnn2d":
            xhat_trans = torch.transpose(
                torch.squeeze(xhat), 3, 0
            )  # [1, Z, X, Y, Real/Imagenary]
            xhat_concat = torch.cat([xhat_trans[0, ...], xhat_trans[1, ...]], 0)
            xhat_expand = xhat_concat.unsqueeze(3)
            phase_regularize_value = self.__dncnn_2d(xhat_expand, reuse=reuse)
            absorption_regularize_value = torch.tensor(0.0)
        else:
            raise NotImplementedError

        if FLAGS.tv3d_z_reg_weight != 0:
            tv_z = self.__total_variation_z(xhat[..., 0])
            tv_z += self.__total_variation_z(xhat[..., 1])
        else:
            tv_z = torch.tensor(0.0)

        # final loss
        loss = (
            mse
            + FLAGS.regularize_weight
            * (absorption_regularize_value + phase_regularize_value)
            + FLAGS.tv3d_z_reg_weight * tv_z
        )

        return (
            loss,
            mse,
            phase_regularize_value,
            absorption_regularize_value,
            xhat,
            Hxhat,
            y,
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
            in_node = F.relu(in_node(input))
            # composite convolutional layers
            for layer in range(2, layer_num):
                in_node = nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size//2, bias=False)
                in_node = F.relu(nn.BatchNorm2d(feature_root)(in_node))
            # output layer and residual learning
            in_node = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size//2)
            output = input - in_node
        return output

    def __dncnn_2d(self, images):  # [N, H, W, C]
        """
        DnCNN as 2.5 dimensional denoiser based on l-2 norm
        """
        a_min = FLAGS.DnCNN_normalization_min
        a_max = FLAGS.DnCNN_normalization_max
        normalized = (images - a_min) / (a_max - a_min)
        denoised = self.__dncnn_inference(torch.clamp(normalized, 0, 1))
        denormalized = denoised * (a_max - a_min) + a_min
        dncnn_res = torch.sum(denormalized**2)
        return dncnn_res
    #########################################
    ###    Parallel & Serial Training     ###
    #########################################


    def __parallelization(self):
        # set up placeholder
        self.lr = torch.tensor([], dtype=torch.float32)
        # Setup placeholder and variables
        # self.Hreal_update_placeholder = torch.empty(
        #     [None, None, None, None, None], dtype=torch.complex64
        # )
        # self.Himag_update_placeholder = torch.empty(
        #     [None, None, None, None, None], dtype=torch.complex64
        # )
        self.Xs = torch.empty(
            [NUM_GPUS, None, 3], dtype=torch.float32
        )
        self.Ys = torch.empty(
            [NUM_GPUS, None, None, None], dtype=torch.float32
        )
        self.Ps = torch.empty(
            [NUM_GPUS, 5, 2], dtype=torch.int32
        )
        self.Ms = torch.empty(
            [NUM_GPUS, None, None], dtype=torch.float32
        )

        # set up optimizer
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # calculate the gradients for each model tower.
        tower_xhat = []
        tower_Hxhat = []
        tower_H = []
        tower_grads = []
        total_loss = torch.tensor(0, dtype=torch.float32)
        total_mse = torch.tensor(0, dtype=torch.float32)
        total_phase = torch.tensor(0, dtype=torch.float32)
        total_absor = torch.tensor(0, dtype=torch.float32)
        reuse_var = False
        for i in range(NUM_GPUS):
            # load H for each tower
            load_H_op, Hreal, Himag = self.__load_H(reuse_var)
            # keep track of load_op_H accross all towers
            tower_H.append(load_H_op)
            # define the loss for each tower.
            (
                loss,
                mse,
                phase_regularize_value,
                absorption_regularize_value,
                xhat,
                Hxhat,
                y,
            ) = self.__tower_loss(i, Hreal, Himag, reuse=reuse_var)
            total_loss = total_loss + loss
            total_mse = total_mse + mse
            total_phase = total_phase + phase_regularize_value
            total_absor = total_absor + absorption_regularize_value
            tower_xhat.append(xhat)
            tower_Hxhat.append(Hxhat)
            # calculate the gradients for the batch of data on this tower.
            grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            # keep track of the gradients across all towers.
            tower_grads.append(grads)
            # reuse variable
            reuse_var = True
        # load H operation
        H_op = torch.stack(tower_H)
        # we must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = self.__average_gradients(tower_grads)
        # apply the gradients to adjust the shared variables.
        opt.step()
        # collect statistics
        statistics_op = [
            total_loss / NUM_GPUS,
            total_mse / NUM_GPUS,
            total_phase / NUM_GPUS,
            total_absor / NUM_GPUS,
        ]
        # keep track of xhat & Hxhat obtained by the last tower
        Hxhat_summary = Hxhat[:2]
        chopped_ground_truth_summary = y[:2]
        # summary operation
        summary_op = (Hxhat_summary, chopped_ground_truth_summary)
        # xhat operation
        xhat_op = tuple(tower_xhat)
        # Hxhat operation
        Hxhat_op = tuple(tower_Hxhat)
        # create a saver
        self.saver = torch.save(self.state_dict(), "{}/{}".format("infer_y", NET_SCOPE))

        return H_op, opt, statistics_op, summary_op, xhat_op, Hxhat_op

    def __load_H(self, reuse):
        # build a graph for loading H (this is due to efficiency)
        with torch.no_grad():
            # set up operation for loading H
            z = torch.zeros((24, 1, 52, 2, 2))
            Hreal = torch.tensor(
                self.Hreal_update_placeholder,
                requires_grad=False,
                dtype=torch.complex64,
            )
            self.H_debug = Hreal
            Himag = torch.tensor(
                self.Himag_update_placeholder,
                requires_grad=False,
                dtype=torch.complex64,
            )
            Hreal.data = self.Hreal_update_placeholder
            Himag.data = self.Himag_update_placeholder
            load_H_op = (Hreal, Himag)
        return load_H_op, Hreal, Himag

def __average_gradients(self, tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    # single GPU collect gradients to cpu (inefficient)
    if len(tower_grads) == 1:
        return tower_grads[0]

    # muliple GPU
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = g.unsqueeze(0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = torch.stack(grads, 0)
        grad = torch.mean(grad, dim=0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

    #########################
    ###    Prediction     ###
    #########################

    # def predict(self, model_path, mesh_grid, Hreal=None, Himag=None):
    #     """Perform the inference of MLP
    #     Args:
    #         model_path: path to the saved model.
    #         mesh_grid:
    #         Hreal: phase light transfer function.
    #         Himag: absorption light transfer function.
    #     Returns:
    #         xhat: final reconstruction.
    #     """
    #     z, x, y, _ = mesh_grid.shape
    #
    #     # placeholder
    #     xhat = np.zeros((z, x, y, 2))
    #     Hxhat = np.zeros((2, x, y))
    #
    #     # Fourier transform
    #     F = lambda x: np.fft.fft2(x)
    #     iF = lambda x: np.fft.ifft2(x)
    #
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     self.to(device)
    #
    #     # load model
    #     self.restore(model_path)
    #
    #     # Start
    #     for start_layer in range(
    #         0, mesh_grid.shape[0], FLAGS.prediction_batch_size
    #     ):
    #         # input mesh grid
    #         partial_mesh_grid = mesh_grid[
    #             start_layer : start_layer + FLAGS.prediction_batch_size
    #         ]
    #         reshaped_mesh_grid = torch.unsqueeze(
    #             torch.reshape(partial_mesh_grid, (-1, 3)), 0
    #         )
    #         # switch based on Hreal & Himag
    #         if Hreal is not None and Himag is not None:
    #             # extract H's
    #             partial_Hreal = Hreal[
    #                 :, start_layer : start_layer + FLAGS.prediction_batch_size
    #             ]
    #             partial_Himag = Himag[
    #                 :, start_layer : start_layer + FLAGS.prediction_batch_size
    #             ]
    #             partial_xhat = self.__tower_loss(
    #                 0, partial_Hreal, partial_Himag, reuse=False
    #             )[4]
    #             Hxhat += torch.sum(
    #                 torch.real(
    #                     iF(
    #                         torch.mul(partial_Hreal, F(partial_xhat[..., 0]))
    #                         + torch.mul(
    #                             partial_Himag, F(partial_xhat[..., 1])
    #                         )
    #                     )
    #                 ),
    #                 dim=1,
    #             )
    #         else:
    #             partial_xhat = self.__tower_loss(
    #                 0, reshaped_mesh_grid, None, reuse=False
    #             )[4]
    #         xhat[
    #             start_layer : start_layer + FLAGS.prediction_batch_size
    #         ] = partial_xhat[0][0]
    #     return Hxhat, xhat
