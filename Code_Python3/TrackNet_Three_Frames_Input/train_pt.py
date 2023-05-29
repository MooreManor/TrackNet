from LoadBatches import TennisDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from Models.TrackNet import TrackNet_pt
import argparse
import torch.nn.functional as F
import os
import cv2
from torchvision.utils import make_grid


# --save_weights_path=weights/model --training_images_name="training_model3_mine.csv" --epochs=100 --n_classes=256 --input_height=360 --input_width=640 --batch_size=2
# --load_weights=2 --step_per_epochs=200
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

device = 'cuda'

def train_summaries(vis, epoch, step):
    rend_imgs = []
    for i in range(vis['vis_pred'].shape[0]):
        vis_input = vis['vis_input'][i]
        vis_pred = vis['vis_pred'][i]
        vis_output = vis['vis_output'][i]
        rend_imgs.append(torch.from_numpy(vis_input).permute(2, 0, 1))
        rend_imgs.append(torch.from_numpy(vis_pred).permute(2, 0, 1))
        rend_imgs.append(torch.from_numpy(vis_output).permute(2, 0, 1))
    images_pred = make_grid(rend_imgs, nrow=3)
    images_pred = images_pred.numpy().transpose(1, 2, 0)
    save_dir = os.path.join('logs', 'train_output_images')
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(save_dir, f'result_epoch{epoch:02d}_step{step:05d}.png'),
        cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)
    )

tennis_dt = TennisDataset(images_path=training_images_name, n_classes=n_classes, input_height=input_height, input_width=input_width,
                          # output_height=input_height, output_width=input_width, num_images=64)
                          output_height=input_height, output_width=input_width)

data_loader = DataLoader(tennis_dt, batch_size=train_batch_size, shuffle=True, num_workers=8)
# data_loader = DataLoader(tennis_dt, batch_size=train_batch_size, shuffle=False, num_workers=8)
net = TrackNet_pt(n_classes=n_classes, input_height=input_height, input_width=input_width).to(device)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
optimizer = torch.optim.Adadelta(net.parameters(), lr=1.0)
# optimizer = torch.optim.Adam(net.parameters(), lr=1.0)

# pbar = tqdm(data_loader,
#                  total=len(data_loader),
#                  desc='Epoch 0')
# criterion = torch.nn.CrossEntropyLoss(reduction='sum')
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.MSELoss()
for epoch in range(epochs):
    pbar = tqdm(data_loader,
                total=len(data_loader),
                desc=f'Epoch {epoch}')
    for step, batch in enumerate(pbar):
        # pass
        input, label, vis_output = batch
        input = input.to(device)
        label = label.to(device)
        pred = net(input)
        import tensorflow as tf
        import numpy as np
        # tmp_label = np.array(label.cpu().detach().numpy())
        # tmp_pred = np.array(pred.cpu().detach().numpy())
        # tf.keras.losses.categorical_crossentropy(np.array(label), np.array(pred))
        # tf.keras.losses.categorical_crossentropy(tmp_label, tmp_pred)
        # loss = pt_categorical_crossentropy(pred, label)
        # loss = F.cross_entropy(pred.reshape(-1, pred.shape[2]), torch.argmax(label, dim=2).reshape(-1))
        loss = criterion(pred.reshape(train_batch_size, input_height, input_width, n_classes).permute(0, 3, 1, 2), torch.argmax(label, dim=2).reshape(train_batch_size, input_height, input_width))
        # loss = criterion(pred, label.float())
        pbar.set_postfix({"loss": float(loss.cpu().detach().numpy())})
        loss.backward()
        optimizer.step()
        if step % 1 == 0:
            vis_input = input.permute(0, 2, 3, 1)[:, :, :, 0:3].cpu().detach().numpy().astype(np.uint8)
            vis_pred = pred.reshape(pred.shape[0], input_height, input_width, n_classes)
            vis_pred = torch.argmax(vis_pred, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 3).cpu().detach().numpy().astype(np.uint8)
            vis_output = np.array(vis_output).astype(np.uint8)
            vis = {'vis_input': vis_input,
                      'vis_pred': vis_pred,
                      'vis_output': vis_output}
            train_summaries(vis, epoch=epoch, step=step)

    if epoch % 1 == 0:
        torch.save(net.state_dict(), save_weights_path + ".0")
    pbar.close()



