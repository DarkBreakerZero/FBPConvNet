import os
from datasets import npz_proj_img_reader_func
import scipy
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import time
import torch.optim as optim
import argparse

from utils import recon_ops
from models import *
from utils import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(train_loader, model, optimizer, scheduler, writer, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    for data in train_loader:


        hdCT = data["hdct"]
        ldCT = data["ldct"]

        hdCT = hdCT.cuda()
        ldCT = ldCT.cuda()

        img_net = model(ldCT)

        loss = F.mse_loss(img_net, hdCT)

        losses.update(loss.item(), hdCT.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalars('train loss', {'mse loss': losses.avg}, epoch + 1)
    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch + 1)

    writer.add_image('train img/label-fbp-result img', normalization(torch.cat([hdCT[0, :, :, :], ldCT[0, :, :, :], img_net[0, :, :, :]], 2)), epoch + 1)
    writer.add_image('train img/residual img', normalization(torch.abs(hdCT[0, :, :, :] - img_net[0, :, :, :])), epoch + 1)
    scheduler.step()

    print('Train Epoch: {}\t train_mse_loss: {:.6f}\t'.format(epoch + 1, losses.avg))

def valid(valid_loader, model, writer, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()

    for data in valid_loader:

        hdCT = data["hdct"]
        ldCT = data["ldct"]

        hdCT = hdCT.cuda()
        ldCT = ldCT.cuda()

        with torch.no_grad():

            img_net = model(ldCT)
            loss = F.mse_loss(img_net, hdCT)

        losses.update(loss.item(), hdCT.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    writer.add_scalars('valid loss', {'mse loss': losses.avg}, epoch+1)
    writer.add_image('valid img/label-fbp-result img', normalization(torch.cat([hdCT[0, :, :, :], ldCT[0, :, :, :], img_net[0, :, :, :]], 2)), epoch + 1)
    writer.add_image('valid img/residual img', normalization(torch.abs(hdCT[0, :, :, :] - img_net[0, :, :, :])), epoch + 1)

    print('Valid Epoch: {}\t valid_mae_loss: {:.6f}\t'.format(epoch + 1, losses.avg))

if __name__ == "__main__":

    cudnn.benchmark = True

    batch_size = 8
    views = 576
    sparse_rate = 6

    method = 'FBPConvNet_LR_V' + str(views//sparse_rate)
    result_path = './runs/' + method + '/logs/'
    save_dir = './runs/' + method + '/checkpoints/'

    # Get dataset
    train_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./train_clear_list_s1e6_v' + str(views//sparse_rate) + '.txt')
    # train_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./valid_clear_list_s1e6_v' + str(views//sparse_rate) + '.txt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)

    valid_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./valid_clear_list_s1e6_v' + str(views//sparse_rate) + '.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=True)

    model = FBPConvNet(model_chl=64)

    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9541)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 100], gamma=0.1)
    #
    if os.path.exists(save_dir) is False:

        model = model.cuda()

    else:
        checkpoint_latest = torch.load(find_lastest_file(save_dir))
        model = load_model(model, checkpoint_latest).cuda()
        optimizer.load_state_dict(checkpoint_latest['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_latest['lr_scheduler'])
        print('Latest checkpoint {0} loaded.'.format(find_lastest_file(save_dir)))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("*"*20 + "Start Train" + "*"*20)

    for epoch in range(0, 100):

        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)

        train(train_loader, model, optimizer, scheduler, writer, epoch)
        valid(valid_loader, model, writer, epoch)

        save_model(model, optimizer, epoch + 1, save_dir, scheduler)