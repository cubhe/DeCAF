import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import time
import gc
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from model.loss import Loss
from model.model_torth_new import DeCAF

NUM_GPUS = torch.cuda.device_count()

def save(model, directory, epoch=None, train_provider=None):
    if epoch is not None:
        directory = os.path.join(directory, "{}_model/".format(epoch))
    else:
        directory = os.path.join(directory, "latest/".format(epoch))
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, "model")
    if train_provider is not None:
        train_provider.save(directory)
    torch.save(model.state_dict(), path)
    print("saved to {}".format(path))
    return path

def restore(model,model_path):
    model.load_state_dict(torch.load(model_path))

def train(FLAGS, train_provider, learning_rate=1e-5, epochs=80):
    output_path=FLAGS.model_save_dir
    # set default device
    print("torch.cuda.is_available()",torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    # initialize trainable variables
    if FLAGS.num_measurements_per_batch <= 0:
        Himag_update_placeholder = train_provider.Himag
        Hreal_update_placeholder = train_provider.Hreal
    else:
        Hreal, Himag = train_provider.sample_partial_tf(FLAGS.num_measurements_per_batch)
        Himag_update_placeholder = Hreal
        Hreal_update_placeholder = Himag

    print("********************")
    # Global Steps
    iters_per_epoch = FLAGS.iters_per_epoch
    global_step = FLAGS.start_epoch * iters_per_epoch

    # initialize summary writer
    summary_writer = SummaryWriter(FLAGS.tf_summary_dir)

    # load previous model if start epoch > 0
    if FLAGS.start_epoch > 0:
        train_provider.restore("{}/latest/".format(output_path))
        restore(os.path.join(output_path, "latest/model"))

    model=DeCAF(FLAGS)
    model.to(device)
    loss=Loss()
    loss.to(device)

    # main loop
    print("Training Started")
    total_time = 0
    for epoch in range(FLAGS.start_epoch, epochs):
        current_time = time.perf_counter()

        # extract learning rate
        if type(learning_rate) is np.ndarray or type(learning_rate) is list:
            lr = learning_rate[epoch]
        elif type(learning_rate) is float:
            lr = learning_rate
        else:
            print("Learning rate should be a list of double or a double scalar.")
            quit()
        summary_writer.add_scalar("learning_rate", lr, global_step)

        # load data
        Xs, Ys, Ps, Ms = train_provider.next_batch_multigpu(NUM_GPUS)

        # iteration
        for iter in range(iters_per_epoch):
            # training
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()

            tower_idx=0

            #Xs, Ys, Ps, Ms = train_provider.next_batch_multigpu(NUM_GPUS)

            x = torch.tensor(Xs[tower_idx, ...]).to(torch.float32).to(device).cuda()
            y = torch.tensor(Ys[tower_idx, ...]).to(torch.float32).to(device).cuda()
            padding = Ps[tower_idx, ...]
            mask = torch.tensor(Ms[tower_idx]).to(torch.float32).to(device).cuda()
            Hreal_update_placeholder = torch.tensor(Hreal_update_placeholder).to(torch.float32).to(device).cuda()
            Himag_update_placeholder = torch.tensor(Himag_update_placeholder).to(torch.float32).to(device).cuda()

            Hxhats,xhats=model(x, Hreal_update_placeholder, Himag_update_placeholder, padding, mask)

            losses,mse,ph_reg_temp,ab_reg_temp=loss(FLAGS, y/2, xhats,y)

            ph_reg = torch.tensor(ph_reg_temp)
            ab_reg = torch.tensor(ab_reg_temp)

            losses.backward()
            optimizer.step()

            if iter % FLAGS.log_iter == 0:
                print(
                    "[Global Step {}] [Epoch {}: {}/{}] [Total = {}] [MSE = {}] "
                    "[Ph_Reg = {}] [Ab_Reg = {}]".format(
                        global_step,
                        epoch + 1,
                        iter + 1,
                        iters_per_epoch,
                        losses.item(),
                        mse.item(),
                        ph_reg.item(),
                        ab_reg.item(),
                        #ph_reg,
                        #ab_reg,
                    )
                )

            # record total loss, mse, ph_reg & ab_reg
            summary_writer.add_scalar("total_loss", losses.item(), global_step)
            summary_writer.add_scalar("mse", mse.item(), global_step)
            summary_writer.add_scalar("phase_reg", ph_reg.item(), global_step)
            summary_writer.add_scalar("abs_reg", ab_reg.item(), global_step)

            # global_step ++
            global_step += 1

        # collect memory
        gc.collect()

        # update provider
        idx = 0
        xhats=xhats.cpu().detach().numpy()
        Hxhats = Hxhats.cpu().detach().numpy()
        ##later
        # for xhat, Hxhat in zip(xhats, Hxhats):
        #     current_error, new_error = train_provider.update(
        #         xhat, partial_estimate=Hxhat, tower_idx=idx
        #     )
        #     idx += 1
        # summary_writer.add_scalar("current_error", current_error, epoch + 1)
        # summary_writer.add_scalar("new_error", new_error, epoch + 1)

        # logs
        execution_time = time.perf_counter() - current_time
        total_time += execution_time
        print(
            "**** [Total epoch time: {:0.4f} seconds] "
            "[Average iteration time: {:0.4f} seconds] ****".format(
                execution_time, total_time / (epoch - FLAGS.start_epoch + 1)
            )
        )

        # save model
        if (epoch + 1) % FLAGS.model_save_epoch == 0:
            save(model,output_path, epoch=epoch + 1)
        if (epoch + 1) % FLAGS.intermediate_result_save_epoch == 0:
            save(model,output_path, train_provider=train_provider)
        else:
            save(model,output_path)

        if (
                FLAGS.num_measurements_per_batch > 0
                and (epoch + 1) % train_provider.scheduler.get_total_blocks() == 0
        ):
            Hreal, Himag = train_provider.sample_partial_tf(FLAGS.num_measurements_per_batch)

        print("Training Ends")
