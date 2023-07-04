from LoadBatches import TennisDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from Models.TrackNet import TrackNet_pt
import argparse
import torch.nn.functional as F
import os
import cv2
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from utils.train_utils import get_log_dir, train_summaries, eval_summaries
import logging

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    # loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
    # loss5.data.item(), loss6.data.item()))

    return loss0, loss
# --save_weights_path=weights/model --training_images_name="training_model3_mine.csv" --epochs=100 --n_classes=256 --input_height=360 --input_width=640 --batch_size=2
# --save_weights_path=weights/model --training_images_name="training_model3_mine.csv" --epochs=100 --n_classes=256 --input_height=540 --input_width=960 --batch_size=2
#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--training_images_name", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 360  )
parser.add_argument("--input_width", type=int , default = 640 )
parser.add_argument("--epochs", type = int, default = 1000 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "-1")
# parser.add_argument("--step_per_epochs", type = int, default = 200 )

args = parser.parse_args()
training_images_name = args.training_images_name
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
# step_per_epochs = args.step_per_epochs
step_per_epochs = 100
train_log_frep = 50
eval_log_frep = 200
eval_freq = 1

device = 'cuda'


log_dir = get_log_dir(exp_name='uiu')
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=log_dir+'/my.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
# summary_writer = SummaryWriter(log_dir)
tennis_dt = TennisDataset(images_path=training_images_name, n_classes=n_classes, input_height=input_height, input_width=input_width,
                          # output_height=input_height, output_width=input_width, num_images=64)
                          output_height=input_height, output_width=input_width)
eval_dt = TennisDataset(images_path='eval.csv', n_classes=n_classes, input_height=input_height, input_width=input_width,
                          # output_height=input_height, output_width=input_width, num_images=64)
                          output_height=input_height, output_width=input_width)

data_loader = DataLoader(tennis_dt, batch_size=train_batch_size, shuffle=True, num_workers=0)
eval_data_loader = DataLoader(eval_dt, batch_size=train_batch_size, shuffle=False, num_workers=8)
# data_loader = DataLoader(tennis_dt, batch_size=train_batch_size, shuffle=True, num_workers=0)
# data_loader = DataLoader(tennis_dt, batch_size=train_batch_size, shuffle=False, num_workers=8)
# data_loader = DataLoader(tennis_dt, batch_size=train_batch_size, shuffle=False, num_workers=0)
# net = TrackNet_pt(n_classes=n_classes, input_height=input_height, input_width=input_width).to(device)
import torch
from Models.uiunet import UIUNET
net = UIUNET(9, 1).to(device)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# optimizer 1 lr0.01 adam
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# optimizer 2 lr0.005 adam
# optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
# optimizer 3 lr0.001 adam
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# optimizer 4 lr1.0 adadelta
optimizer = torch.optim.Adadelta(net.parameters(), lr=1.0)
# optimizer = torch.optim.Adam(net.parameters(), lr=1.0)

# pbar = tqdm(data_loader,
#                  total=len(data_loader),
#                  desc='Epoch 0')
# criterion = torch.nn.CrossEntropyLoss(reduction='sum')
# criterion = torch.nn.CrossEntropyLoss()
criterion = nn.BCELoss(size_average=True)
# criterion = torch.nn.MSELoss()

#---------------------------------------------------------------------
# single sample overfit
# batch = next(iter(data_loader))
# input, label, vis_output = batch
# for epoch in range(epochs):
#     input = input.to(device)
#     label = label.to(device)
#     d0, d1, d2, d3, d4, d5, d6 = net(input)
#     pred = d0
#     loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label.reshape(label.shape[0], 1, input_height, input_width))
#     # pbar.set_postfix({"loss": float(loss.cpu().detach().numpy())})
#     print(f'{epoch}/{epochs}', loss)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#             # if step % log_frep == 0:
#     vis_input = input.permute(0, 2, 3, 1)[:, :, :, 0:3].cpu().detach().numpy().astype(np.uint8)
#     # vis_pred = pred.reshape(pred.shape[0], input_height, input_width, n_classes)
#     vis_pred = pred.reshape(pred.shape[0], input_height, input_width, 1)*255.
#     # vis_pred = torch.argmax(vis_pred, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 3).cpu().detach().numpy().astype(np.uint8)
#     vis_pred = vis_pred.repeat(1, 1, 1, 3).cpu().detach().numpy().astype(np.uint8)
#     vis_output = np.array(vis_output).astype(np.uint8)
#     vis = {'vis_input': vis_input,
#               'vis_pred': vis_pred,
#               'vis_output': vis_output}
#     train_summaries(vis, epoch=epoch, step=0, log_dir=log_dir)

# #---------------------------------------------------------------------
# # train
best_acc = 100000
for epoch in range(epochs):
    pbar = tqdm(data_loader,
                # total=len(data_loader),
                total=len(data_loader) if step_per_epochs > len(data_loader) else step_per_epochs,
                desc=f'Epoch {epoch}')
    net.train()
    for step, batch in enumerate(pbar):
        # pass
        if step >= step_per_epochs:
            break
        input, label, vis_output = batch
        input = input.to(device)
        label = label.to(device)
        d0, d1, d2, d3, d4, d5, d6 = net(input)
        pred = d0
        # loss = criterion(pred, label.reshape(label.shape[0], -1, 1))
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label.reshape(label.shape[0], 1, input_height, input_width))
        # loss = criterion(pred, label.float())
        pbar.set_postfix({"loss": float(loss.cpu().detach().numpy())})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % train_log_frep == 0:
            vis_input = input.permute(0, 2, 3, 1)[:, :, :, 0:3].cpu().detach().numpy().astype(np.uint8)
            # vis_pred = pred.reshape(pred.shape[0], input_height, input_width, n_classes)
            vis_pred = pred.reshape(pred.shape[0], input_height, input_width, 1)*255.
            # vis_pred = torch.argmax(vis_pred, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 3).cpu().detach().numpy().astype(np.uint8)
            vis_pred = vis_pred.repeat(1, 1, 1, 3).cpu().detach().numpy().astype(np.uint8)
            vis_output = np.array(vis_output).astype(np.uint8)
            vis = {'vis_input': vis_input,
                      'vis_pred': vis_pred,
                      'vis_output': vis_output}
            train_summaries(vis, epoch=epoch, step=step, log_dir=log_dir)
    pbar.close()


    if epoch % eval_freq == 0:
        net.eval()
        eval_pbar = tqdm(eval_data_loader,
                    total=len(eval_data_loader))
        all_loss = 0
        for step, batch in enumerate(eval_pbar):
            input, label, vis_output = batch
            input = input.to(device)
            label = label.to(device)
            with torch.no_grad():
                d1, d2, d3, d4, d5, d6, d7 = net(input)
                pred = d1[:, :, :, :]
            loss = bce_loss(pred, label.reshape(label.shape[0], 1, input_height, input_width))
            all_loss += loss.item()
            if step % eval_log_frep == 0:
                vis_input = input.permute(0, 2, 3, 1)[:, :, :, 0:3].cpu().detach().numpy().astype(np.uint8)
                vis_pred = pred.reshape(pred.shape[0], input_height, input_width, 1) * 255.
                vis_pred = vis_pred.repeat(1, 1, 1, 3).cpu().detach().numpy().astype(np.uint8)
                vis_output = np.array(vis_output).astype(np.uint8)
                vis = {'vis_input': vis_input,
                       'vis_pred': vis_pred,
                       'vis_output': vis_output}
                eval_summaries(vis, epoch=epoch, step=step, log_dir=log_dir)
        logging.info(f"Epoch {epoch}: {all_loss}")
        # logging.info(f"Epoch 0: {all_loss}")

        # summary_writer.add_scalar('val_loss', all_loss, global_step=epoch)
        if best_acc>all_loss:
            best_acc = min(best_acc, all_loss)
            logging.info(f"Best eval loss Found at epoch {epoch}: {best_acc}")
            torch.save(net.state_dict(), save_weights_path + ".pt.0")
        eval_pbar.close()



