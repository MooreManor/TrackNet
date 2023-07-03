import time
import random
import string
import os
import torch
from torchvision.utils import make_grid
import cv2
import logging
logger = logging.getLogger(__name__)

def get_log_dir(exp_name, log_dir='./logs'):
    letters = string.ascii_letters
    # timestamp = datetime.now().strftime('%b%d-%H-%M-%S-') + ''.join(random.choice(letters) for i in range(3))
    timestamp = time.strftime('%d-%m-%Y_%H-%M-%S') + ''.join(random.choice(letters) for i in range(3))
    log_name = timestamp
    log_dir = os.path.join(log_dir, exp_name, log_name)
    logger.info('log name: {}'.format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def train_summaries(vis, epoch, step, log_dir):
    rend_imgs = []
    nb_max_img = min(6, vis['vis_pred'].shape[0])
    for i in range(nb_max_img):
        vis_input = vis['vis_input'][i]
        vis_pred = vis['vis_pred'][i]
        vis_output = vis['vis_output'][i]
        rend_imgs.append(torch.from_numpy(vis_input).permute(2, 0, 1))
        rend_imgs.append(torch.from_numpy(vis_pred).permute(2, 0, 1))
        rend_imgs.append(torch.from_numpy(vis_output).permute(2, 0, 1))
    images_pred = make_grid(rend_imgs, nrow=3)
    images_pred = images_pred.numpy().transpose(1, 2, 0)
    save_dir = os.path.join(log_dir, 'train_output_images')
    # save_dir = os.path.join('logs', 'train_output_images_bs')
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(save_dir, f'result_epoch{epoch:03d}_step{step:05d}.png'),
        cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)
    )

def eval_summaries(vis, epoch, step, log_dir):
    rend_imgs = []
    nb_max_img = min(6, vis['vis_pred'].shape[0])
    for i in range(nb_max_img):
        vis_input = vis['vis_input'][i]
        vis_pred = vis['vis_pred'][i]
        vis_output = vis['vis_output'][i]
        rend_imgs.append(torch.from_numpy(vis_input).permute(2, 0, 1))
        rend_imgs.append(torch.from_numpy(vis_pred).permute(2, 0, 1))
        rend_imgs.append(torch.from_numpy(vis_output).permute(2, 0, 1))
    images_pred = make_grid(rend_imgs, nrow=3)
    images_pred = images_pred.numpy().transpose(1, 2, 0)
    save_dir = os.path.join(log_dir, 'eval_output_images')
    # save_dir = os.path.join('logs', 'train_output_images_bs')
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(save_dir, f'result_epoch{epoch:03d}_step{step:05d}.png'),
        cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)
    )